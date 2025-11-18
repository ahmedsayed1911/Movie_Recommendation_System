# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
from typing import List, Optional, Dict

# semantic / embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

st.set_page_config(page_title="Movie Recommender (Semantic + TMDB)", layout="wide")

# ------------------------------
# Config
# ------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")  # set as Streamlit secret or env var
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"

# Chroma persistence directory (on Streamlit Cloud it's ephemeral but works per deployment)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Embedding model name
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

# ------------------------------
# Load data
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(path="netflixData.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (some files have Title, title, etc.)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # expected Title, Description, Genres, Release Date (or Release_Date)
    # unify column names
    if 'Title' not in df.columns:
        # try lowercase
        if 'title' in df.columns:
            df.rename(columns={'title': 'Title'}, inplace=True)
        else:
            raise ValueError("Data must contain a 'Title' column.")
    # ensure description/genres exist
    for c in ['Description', 'Genres', 'Release Date', 'Release_Date', 'Year']:
        if c not in df.columns:
            df[c] = pd.NA

    # create consistent fields
    df['title'] = df['Title'].astype(str)
    # description
    df['description'] = df['Description'].fillna("").astype(str)
    # genres normalize: try splitting by common separators
    df['genres'] = df['Genres'].fillna("").astype(str)
    # extract year
    def extract_year(row):
        for c in ['Year', 'Release Date', 'Release_Date', 'release_date']:
            v = row.get(c)
            if pd.isna(v):
                continue
            try:
                s = str(v)
                if len(s) >= 4:
                    y = int(s[:4])
                    return y
            except:
                continue
        return np.nan
    df['year'] = df.apply(extract_year, axis=1)
    # combined text for simple TFIDF fallback (if needed)
    df['combined'] = (df['genres'] + " " + df['description']).str.strip().fillna("")
    return df.reset_index(drop=True)

df = load_data("/mnt/data/netflixData.csv")  # path in your repo

# ------------------------------
# Chroma + Embeddings init
# ------------------------------
@st.cache_resource(show_spinner=False)
def init_chroma(client_settings: Optional[dict] = None):
    # Use local persistence
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR)
    client = chromadb.Client(settings=settings)
    return client

chroma_client = init_chroma()

@st.cache_data(show_spinner=False)
def get_embedder(model_name=EMBEDDER_MODEL):
    model = SentenceTransformer(model_name)
    return model

embedder = get_embedder()

# helper: collection name
COLLECTION_NAME = "netflix_movies_v1"

def create_or_get_collection():
    # create or get
    try:
        coll = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        coll = chroma_client.create_collection(name=COLLECTION_NAME)
    return coll

collection = create_or_get_collection()

# If collection empty, populate it from dataframe
@st.cache_data(show_spinner=True)
def ensure_collection_populated(df: pd.DataFrame, collection) -> None:
    # check count
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count and count > 0:
        return
    # prepare metadata and embeddings
    texts = (df['title'].fillna("") + ". " + df['description'].fillna("")).tolist()
    ids = df.index.astype(str).tolist()
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "title": row['title'],
            "genres": row.get('genres', ""),
            "year": int(row['year']) if not pd.isna(row['year']) else None,
            "description": row.get('description', "")
        })
    # compute embeddings in batches
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        embs = embedder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        collection.add(
            documents=[t for t in batch_texts],
            metadatas=batch_meta,
            ids=batch_ids,
            embeddings=embs.tolist()
        )
    # persist
    try:
        chroma_client.persist()
    except Exception:
        pass

ensure_collection_populated(df, collection)

# ------------------------------
# TMDB helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def tmdb_search(title: str):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(TMDB_SEARCH_URL, params={"api_key": TMDB_API_KEY, "query": title, "include_adult": False}, timeout=8)
        r.raise_for_status()
        j = r.json()
        results = j.get("results", [])
        return results[0] if results else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_get_movie(movie_id: int):
    if not TMDB_API_KEY or not movie_id:
        return None
    try:
        r = requests.get(TMDB_MOVIE_URL.format(movie_id), params={"api_key": TMDB_API_KEY}, timeout=8)
        r.raise_for_status()
        details = r.json()
        # get external ids
        r2 = requests.get(TMDB_MOVIE_URL.format(movie_id)+"/external_ids", params={"api_key": TMDB_API_KEY}, timeout=8)
        if r2.ok:
            details['external_ids'] = r2.json()
        return details
    except Exception:
        return None

# ------------------------------
# Semantic search wrapper
# ------------------------------
def semantic_search_by_text(query: str, top_k=10, filters: Optional[Dict]=None):
    # compute embedding
    emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    # build filter dict for chroma
    where = {}
    if filters:
        # chroma expects metadata filters; simple equality checks
        # We'll implement year range via post-filtering
        if 'genres' in filters and filters['genres']:
            # nothing to pass easily; will filter results after retrieval
            pass
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k*3,  # retrieve more to allow post-filtering
        include=['metadatas', 'scores', 'documents', 'ids']
    )
    hits = []
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    scores = results['distances'][0] if 'distances' in results else results.get('scores', [[]])[0]
    for doc, meta, score, id_ in zip(docs, metas, scores, results['ids'][0]):
        # convert chroma distance -> similarity-like (smaller distance is better)
        # We'll use score as-is but sort ascending if it's distance
        hits.append({
            "id": int(id_),
            "title": meta.get('title'),
            "genres": meta.get('genres'),
            "year": meta.get('year'),
            "description": meta.get('description'),
            "score": float(score)
        })
    # post filter by genre/year if given
    if filters:
        if 'genres' in filters and filters['genres']:
            selected = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in selected)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    # sort by score (chroma returns distances; smaller is better) -> ascending
    hits = sorted(hits, key=lambda x: x['score'])[:top_k]
    return hits

def semantic_search_by_title(title: str, top_k=10, filters: Optional[Dict]=None):
    # find title in df
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        # try fuzzy by contains
        fuzzy = df[df['title'].str.lower().str.contains(title.lower())]
        if fuzzy.empty:
            return None
        idx = fuzzy.index[0]
    else:
        idx = matches.index[0]
    # retrieve by id
    id_str = str(idx)
    results = collection.query(
        ids=[id_str],
        n_results=top_k+1,
        include=['metadatas', 'distances', 'documents', 'ids']
    )
    # The query by id returns related items including itself; we'll filter later
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0] if 'distances' in results else results.get('scores', [[]])[0]
    ids = results.get('ids', [[]])[0]
    hits = []
    for doc, meta, d, i in zip(docs, metas, distances, ids):
        if int(i) == int(id_str):
            continue
        hits.append({
            "id": int(i),
            "title": meta.get('title'),
            "genres": meta.get('genres'),
            "year": meta.get('year'),
            "description": meta.get('description'),
            "score": float(d)
        })
    # apply filters similar to semantic_search_by_text
    if filters:
        if 'genres' in filters and filters['genres']:
            selected = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in selected)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    hits = sorted(hits, key=lambda x: x['score'])[:top_k]
    return hits

# ------------------------------
# UI
# ------------------------------
st.title("🎯 Movie Recommender — Semantic Search (Chroma) + Posters (TMDB)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Search mode:", options=["By movie title (semantic)", "By text/query (semantic)"])
    query_input = st.text_input("اكتب اسم الفيلم أو استعلام (title / keywords):")
    top_k = st.slider("عدد النتائج", 5, 30, 10)
    # genres options from df
    all_genres = set()
    for g in df['genres'].dropna().astype(str):
        for sep in ['|', ',', ';']:
            if sep in g:
                parts = [p.strip() for p in g.split(sep) if p.strip()]
                all_genres.update(parts)
                break
        else:
            if g.strip():
                all_genres.add(g.strip())
    genre_list = sorted(list(all_genres))[:200]
    selected_genres = st.multiselect("فلتر حسب النوع (Genres):", options=genre_list)
    years = df['year'].dropna().astype(int) if not df['year'].dropna().empty else pd.Series([2000])
    ymin = int(years.min()) if not years.empty else 1900
    ymax = int(years.max()) if not years.empty else 2025
    year_range = st.slider("فلتر حسب السنة:", min_value=1900, max_value=2025, value=(ymin, ymax))
    use_tmdb = st.checkbox("اجلب Posters و Ratings من TMDB (مطلوب TMDB_API_KEY)", value=True)
    st.markdown("---")
    st.markdown("Secrets needed: set `TMDB_API_KEY` in Streamlit secrets. Chroma persistence dir can be overridden with `CHROMA_PERSIST_DIR` env var.")

# Main action
if st.button("Search"):
    if not query_input or query_input.strip() == "":
        st.warning("اكتب استعلام أو اسم فيلم!")
    else:
        filters = {
            "genres": selected_genres,
            "year_min": year_range[0],
            "year_max": year_range[1]
        }
        if mode == "By movie title (semantic)":
            hits = semantic_search_by_title(query_input, top_k=top_k, filters=filters)
        else:
            hits = semantic_search_by_text(query_input, top_k=top_k, filters=filters)

        if hits is None:
            st.error("لم يتم العثور على الفيلم في الداتا — جرّب كتابة اسم أقرب.")
        elif len(hits) == 0:
            st.info("لا توجد نتائج بعد تطبيق الفلاتر.")
        else:
            st.success(f"Found {len(hits)} results")
            enhanced = []
            for h in hits:
                title = h['title']
                poster_url = None
                tmdb_rating = None
                imdb_link = None
                # attempt TMDB lookup
                if use_tmdb and TMDB_API_KEY:
                    s = tmdb_search(title)
                    if s:
                        tmdb_id = s.get('id')
                        details = tmdb_get_movie(tmdb_id)
                        if details:
                            poster = details.get('poster_path')
                            if poster:
                                poster_url = TMDB_IMAGE_BASE + poster
                            tmdb_rating = details.get('vote_average')
                            imdb_id = details.get('external_ids', {}).get('imdb_id')
                            if imdb_id:
                                imdb_link = f"https://www.imdb.com/title/{imdb_id}"
                cols = st.columns([1, 3])
                with cols[0]:
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                    else:
                        st.write("No poster")
                with cols[1]:
                    st.markdown(f"### {title} ({h.get('year') or '—'})")
                    if tmdb_rating:
                        st.write(f"**TMDB rating:** {tmdb_rating} / 10")
                    if imdb_link:
                        st.write(f"[Open on IMDb]({imdb_link})")
                    st.write(f"**Genres:** {h.get('genres') or '—'}")
                    desc = h.get('description') or ""
                    st.write(desc[:600] + ("..." if len(desc) > 600 else ""))
                    st.write(f"**Chroma distance:** {round(h.get('score', 0), 4)}")
                    st.markdown("---")
                enhanced.append({
                    "title": h['title'],
                    "year": h.get('year'),
                    "genres": h.get('genres'),
                    "description": h.get('description'),
                    "chroma_distance": h.get('score'),
                    "tmdb_rating": tmdb_rating,
                    "poster_url": poster_url,
                    "imdb_link": imdb_link
                })
            # download
            df_out = pd.DataFrame(enhanced)
            csv_buffer = StringIO()
            df_out.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            st.download_button("Download results as CSV", data=csv_data, file_name="semantic_search_results.csv", mime="text/csv")

else:
    st.info("اكتب استعلام واضغط Search. يمكنك التبديل بين البحث بالعنوان أو بالوصف (semantic).")

# footer: quick sample counts
with st.expander("Dataset info"):
    st.write(f"Rows: {len(df)}")
    st.write(f"Unique titles: {df['title'].nunique()}")
    st.write("Sample titles:")
    st.write(df['title'].drop_duplicates().head(50).tolist())
