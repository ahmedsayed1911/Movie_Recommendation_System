# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
from typing import Optional, Dict

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
def load_data(path="/mnt/data/netflixData.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c: c.strip(): c.strip() for c in df.columns} if False else {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    if 'Title' not in df.columns and 'title' in df.columns:
        df.rename(columns={'title': 'Title'}, inplace=True)
    if 'Title' not in df.columns:
        raise ValueError("Data must contain a 'Title' column.")
    for c in ['Description', 'Genres', 'Release Date', 'Release_Date', 'Year']:
        if c not in df.columns:
            df[c] = pd.NA

    df['title'] = df['Title'].astype(str)
    df['description'] = df['Description'].fillna("").astype(str)
    df['genres'] = df['Genres'].fillna("").astype(str)

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
    df['combined'] = (df['genres'] + " " + df['description']).str.strip().fillna("")
    return df.reset_index(drop=True)

# load from repo path (Streamlit Cloud will use the repo root)
try:
    df = load_data("/mnt/data/netflixData.csv")
except Exception:
    # fallback to relative path in repo if you committed the csv
    df = load_data("netflixData.csv")

# ------------------------------
# Chroma + Embeddings init (robust)
# ------------------------------
@st.cache_resource(show_spinner=False)
def init_chroma(persist_dir: str = CHROMA_PERSIST_DIR):
    # ensure dir exists
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        pass

    # Try preferred persistent settings (duckdb+parquet). If chromadb config raises, fall back to default Client()
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        client = chromadb.Client(settings=settings)
        st.info("Chroma: using duckdb+parquet persistence.")
        return client
    except Exception as e:
        st.warning(f"Chroma persistent init failed, falling back to in-memory client. ({e})")
        try:
            client = chromadb.Client()
            return client
        except Exception as e2:
            st.error(f"Failed to init chromadb client: {e2}")
            raise

chroma_client = init_chroma()

@st.cache_data(show_spinner=False)
def get_embedder(model_name=EMBEDDER_MODEL):
    # sentence-transformers will download model on first run (may take time)
    model = SentenceTransformer(model_name)
    return model

embedder = get_embedder()

# helper: collection name
COLLECTION_NAME = "netflix_movies_v1"

def create_or_get_collection():
    try:
        coll = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        coll = chroma_client.create_collection(name=COLLECTION_NAME)
    return coll

collection = create_or_get_collection()

# populate collection if empty
@st.cache_data(show_spinner=True)
def ensure_collection_populated(df: pd.DataFrame, collection) -> None:
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count and count > 0:
        return
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
        r2 = requests.get(TMDB_MOVIE_URL.format(movie_id)+"/external_ids", params={"api_key": TMDB_API_KEY}, timeout=8)
        if r2.ok:
            details['external_ids'] = r2.json()
        return details
    except Exception:
        return None

# ------------------------------
# Semantic search functions
# ------------------------------
def semantic_search_by_text(query: str, top_k=10, filters: Optional[Dict]=None):
    emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k*3,
        include=['metadatas', 'scores', 'documents', 'ids']
    )
    hits = []
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    scores = results.get('distances', results.get('scores', [[]]))[0]
    ids = results['ids'][0]
    for doc, meta, score, id_ in zip(docs, metas, scores, ids):
        hits.append({
            "id": int(id_),
            "title": meta.get('title'),
            "genres": meta.get('genres'),
            "year": meta.get('year'),
            "description": meta.get('description'),
            "score": float(score)
        })
    # post filtering
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

def semantic_search_by_title(title: str, top_k=10, filters: Optional[Dict]=None):
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        fuzzy = df[df['title'].str.lower().str.contains(title.lower())]
        if fuzzy.empty:
            return None
        idx = fuzzy.index[0]
    else:
        idx = matches.index[0]
    id_str = str(idx)
    results = collection.query(
        ids=[id_str],
        n_results=top_k+1,
        include=['metadatas', 'distances', 'documents', 'ids']
    )
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]
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
# UI (same as previous)
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
            df_out = pd.DataFrame(enhanced)
            csv_buffer = StringIO()
            df_out.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            st.download_button("Download results as CSV", data=csv_data, file_name="semantic_search_results.csv", mime="text/csv")
else:
    st.info("اكتب استعلام واضغط Search. يمكنك التبديل بين البحث بالعنوان أو بالوصف (semantic).")

with st.expander("Dataset info"):
    st.write(f"Rows: {len(df)}")
    st.write(f"Unique titles: {df['title'].nunique()}")
    st.write("Sample titles:")
    st.write(df['title'].drop_duplicates().head(50).tolist())
