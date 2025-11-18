# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import traceback

st.set_page_config(page_title="Movie Recommender (Semantic + TMDB)", layout="wide")

# ------------------------------
# Optional imports (graceful)
# ------------------------------
HAS_SENTENCE = True
HAS_CHROMA = True
HAS_SKLEARN = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    HAS_SENTENCE = False

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    HAS_CHROMA = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SKLEARN = False

# ------------------------------
# Config (env / secrets)
# ------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")  # set as Streamlit secret or env var
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

# ------------------------------
# Load data
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(path="/mnt/data/netflixData.csv"):
    df = pd.read_csv(path)
    # strip column names
    df.rename(columns=lambda c: c.strip(), inplace=True)
    # make sure Title exists
    if 'Title' not in df.columns and 'title' in df.columns:
        df.rename(columns={'title': 'Title'}, inplace=True)
    if 'Title' not in df.columns:
        raise ValueError("Data must contain a 'Title' column (Title).")
    # ensure columns exist
    for c in ['Description', 'Genres', 'Release Date', 'Year']:
        if c not in df.columns:
            df[c] = pd.NA
    # uniform fields
    df['title'] = df['Title'].astype(str).fillna("")
    df['description'] = df['Description'].fillna("").astype(str)
    df['genres'] = df['Genres'].fillna("").astype(str)
    # extract year heuristically
    def extract_year(row):
        for c in ['Year', 'Release Date', 'Release_Date', 'release_date']:
            if c in row and pd.notna(row[c]):
                s = str(row[c])
                if len(s) >= 4:
                    try:
                        return int(s[:4])
                    except:
                        continue
        return np.nan
    df['year'] = df.apply(extract_year, axis=1)
    df['combined'] = (df['genres'].astype(str) + " " + df['description'].astype(str)).str.strip()
    return df.reset_index(drop=True)

# try load; if fails, streamlit will show error
try:
    df = load_data("/mnt/data/netflixData.csv")
except Exception as e:
    st.error("فشل تحميل الداتا: " + str(e))
    st.stop()

# ------------------------------
# Embeddings / Chroma / fallback
# ------------------------------
embedder = None
chroma_client = None
collection = None
nn_model = None
embeddings_matrix = None
id_to_index = None

def init_embedder():
    global embedder
    if embedder is not None:
        return embedder
    if not HAS_SENTENCE:
        st.error("المكتبة sentence-transformers غير مثبتة. أضف 'sentence-transformers' إلى requirements.txt ثم deploy ثانية.")
        st.stop()
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    return embedder

def init_chroma():
    global chroma_client, collection
    if not HAS_CHROMA:
        return None
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR)
        chroma_client = chromadb.Client(settings=settings)
        # create/get collection
        try:
            collection = chroma_client.get_collection(name="netflix_movies_v1")
        except Exception:
            collection = chroma_client.create_collection(name="netflix_movies_v1")
        return chroma_client
    except Exception as e:
        # return None -> fallback
        st.warning("تعذّر تهيئة chromadb — سيتم التحول لِـ in-memory semantic search. (يمكن أن يكون السبب اختلاف نسخة chromadb أو صلاحيات الملف).")
        # print traceback to logs area (not too verbose)
        st.write(f"Debug (chroma init): {e}")
        return None

def populate_chroma_from_df():
    global collection
    if chroma_client is None or collection is None:
        return False
    try:
        # check if already populated
        try:
            cnt = collection.count()
        except Exception:
            cnt = 0
        if cnt and cnt > 0:
            return True
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
        model = init_embedder()
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            embs = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            # chroma expects list of floats
            collection.add(documents=batch_texts, metadatas=batch_meta, ids=batch_ids, embeddings=embs.tolist())
        try:
            chroma_client.persist()
        except Exception:
            pass
        return True
    except Exception as e:
        st.warning("خطأ أثناء تعبئة chroma: " + str(e))
        return False

def build_inmemory_embeddings_and_nn():
    global embeddings_matrix, nn_model, id_to_index
    if not HAS_SENTENCE:
        st.error("sentence-transformers غير مثبت — لا يمكن بناء الـ embeddings.")
        st.stop()
    model = init_embedder()
    texts = (df['title'].fillna("") + ". " + df['description'].fillna("")).tolist()
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings_matrix = embs
    id_to_index = {str(i): i for i in range(len(texts))}
    if HAS_SKLEARN:
        nn_model_local = NearestNeighbors(n_neighbors=50, metric="cosine")
        nn_model_local.fit(embs)
        nn_model = nn_model_local
    else:
        nn_model = None

# Try init chroma; if fails fallback to in-memory
if HAS_CHROMA:
    try:
        chroma_client = init_chroma()
        if chroma_client is not None:
            populate_chroma_from_df()
        else:
            build_inmemory_embeddings_and_nn()
    except Exception:
        build_inmemory_embeddings_and_nn()
else:
    build_inmemory_embeddings_and_nn()

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
def semantic_search_by_text_chroma(query: str, top_k=10, filters=None):
    # uses chroma collection
    if collection is None:
        return []
    model = init_embedder()
    emb = model.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k*3, include=['metadatas', 'distances', 'ids'])
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    dists = results.get('distances', [[]])[0] if 'distances' in results else results.get('scores', [[]])[0]
    ids = results.get('ids', [[]])[0]
    hits = []
    for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
        hits.append({
            "id": int(id_),
            "title": meta.get('title'),
            "genres": meta.get('genres'),
            "year": meta.get('year'),
            "description": meta.get('description'),
            "score": float(dist)
        })
    # apply basic filters
    if filters:
        if 'genres' in filters and filters['genres']:
            sel = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in sel)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    hits = sorted(hits, key=lambda x: x['score'])[:top_k]
    return hits

def semantic_search_by_title_chroma(title: str, top_k=10, filters=None):
    # query by id (first find matching df row)
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        fuzzy = df[df['title'].str.lower().str.contains(title.lower())]
        if fuzzy.empty:
            return None
        idx = fuzzy.index[0]
    else:
        idx = matches.index[0]
    id_str = str(idx)
    results = collection.query(ids=[id_str], n_results=top_k+5, include=['metadatas', 'distances', 'ids'])
    metas = results.get('metadatas', [[]])[0]
    dists = results.get('distances', [[]])[0] if 'distances' in results else results.get('scores', [[]])[0]
    ids = results.get('ids', [[]])[0]
    hits = []
    for meta, d, i in zip(metas, dists, ids):
        if str(i) == id_str:
            continue
        hits.append({
            "id": int(i),
            "title": meta.get('title'),
            "genres": meta.get('genres'),
            "year": meta.get('year'),
            "description": meta.get('description'),
            "score": float(d)
        })
    # apply filters
    if filters:
        if 'genres' in filters and filters['genres']:
            sel = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in sel)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    hits = sorted(hits, key=lambda x: x['score'])[:top_k]
    return hits

def semantic_search_inmemory_title(title: str, top_k=10, filters=None):
    # use nn_model or brute-force cosine if nn not available
    global embeddings_matrix, nn_model
    if embeddings_matrix is None:
        build_inmemory_embeddings_and_nn()
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        fuzzy = df[df['title'].str.lower().str.contains(title.lower())]
        if fuzzy.empty:
            return None
        idx = fuzzy.index[0]
    else:
        idx = matches.index[0]
    q_emb = embeddings_matrix[int(idx)].reshape(1, -1)
    if nn_model is not None:
        dists, neigh_idxs = nn_model.kneighbors(q_emb, n_neighbors=min(50, len(embeddings_matrix)))
        neigh_idxs = neigh_idxs[0]
        hits = []
        for ni in neigh_idxs:
            if ni == idx:
                continue
            hits.append({
                "id": int(ni),
                "title": df.loc[ni, 'title'],
                "genres": df.loc[ni, 'genres'],
                "year": int(df.loc[ni, 'year']) if not pd.isna(df.loc[ni, 'year']) else None,
                "description": df.loc[ni, 'description'],
                "score": float(dists[0][list(neigh_idxs).index(ni)])
            })
    else:
        # brute force cosine
        sims = cosine_similarity(q_emb, embeddings_matrix)[0]
        idxs = np.argsort(-sims)[:50]
        hits = []
        for ni in idxs:
            if ni == idx:
                continue
            hits.append({
                "id": int(ni),
                "title": df.loc[ni, 'title'],
                "genres": df.loc[ni, 'genres'],
                "year": int(df.loc[ni, 'year']) if not pd.isna(df.loc[ni, 'year']) else None,
                "description": df.loc[ni, 'description'],
                "score": float(1.0 - sims[ni])  # pseudo-distance
            })
    # apply filters
    if filters:
        if 'genres' in filters and filters['genres']:
            sel = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in sel)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    hits = hits[:top_k]
    return hits

def semantic_search_inmemory_text(query: str, top_k=10, filters=None):
    global embeddings_matrix, nn_model
    if embeddings_matrix is None:
        build_inmemory_embeddings_and_nn()
    model = init_embedder()
    q_emb = model.encode([query], convert_to_numpy=True)[0].reshape(1, -1)
    if nn_model is not None:
        dists, neigh_idxs = nn_model.kneighbors(q_emb, n_neighbors=min(50, len(embeddings_matrix)))
        neigh_idxs = neigh_idxs[0]
        hits = []
        for ni in neigh_idxs:
            hits.append({
                "id": int(ni),
                "title": df.loc[ni, 'title'],
                "genres": df.loc[ni, 'genres'],
                "year": int(df.loc[ni, 'year']) if not pd.isna(df.loc[ni, 'year']) else None,
                "description": df.loc[ni, 'description'],
                "score": float(dists[0][list(neigh_idxs).index(ni)])
            })
    else:
        sims = cosine_similarity(q_emb, embeddings_matrix)[0]
        idxs = np.argsort(-sims)[:50]
        hits = []
        for ni in idxs:
            hits.append({
                "id": int(ni),
                "title": df.loc[ni, 'title'],
                "genres": df.loc[ni, 'genres'],
                "year": int(df.loc[ni, 'year']) if not pd.isna(df.loc[ni, 'year']) else None,
                "description": df.loc[ni, 'description'],
                "score": float(1.0 - sims[ni])
            })
    # apply filters
    if filters:
        if 'genres' in filters and filters['genres']:
            sel = [g.strip().lower() for g in filters['genres']]
            hits = [h for h in hits if any(g in (h.get('genres') or "").lower() for g in sel)]
        if 'year_min' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') >= filters['year_min'])]
        if 'year_max' in filters:
            hits = [h for h in hits if (h.get('year') is None) or (h.get('year') <= filters['year_max'])]
    hits = hits[:top_k]
    return hits

# ------------------------------
# UI
# ------------------------------
st.title("🎯 Movie Recommender — Semantic Search (Chroma fallback) + Posters (TMDB)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Search mode:", options=["By movie title (semantic)", "By text/query (semantic)"])
    query_input = st.text_input("اكتب اسم الفيلم أو استعلام (title / keywords):")
    top_k = st.slider("عدد النتائج", 5, 30, 10)
    # genres options
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
    genre_list = sorted(list(all_genres))
    selected_genres = st.multiselect("فلتر حسب النوع (Genres):", options=genre_list)
    years = df['year'].dropna().astype(int) if not df['year'].dropna().empty else pd.Series([2000])
    ymin = int(years.min()) if not years.empty else 1900
    ymax = int(years.max()) if not years.empty else 2025
    year_range = st.slider("فلتر حسب السنة:", min_value=1900, max_value=2025, value=(ymin, ymax))
    use_tmdb = st.checkbox("اجلب Posters و Ratings من TMDB (مطلوب TMDB_API_KEY)", value=True)
    st.markdown("---")
    st.markdown("Secrets needed: set `TMDB_API_KEY` in Streamlit secrets. If chromadb fails, the app uses an in-memory semantic fallback.")

if st.button("Search"):
    if not query_input or query_input.strip() == "":
        st.warning("اكتب استعلام أو اسم فيلم!")
    else:
        filters = {"genres": selected_genres, "year_min": year_range[0], "year_max": year_range[1]}
        try:
            if chroma_client is not None and collection is not None:
                if mode.startswith("By movie"):
                    hits = semantic_search_by_title_chroma(query_input, top_k=top_k, filters=filters)
                else:
                    hits = semantic_search_by_text_chroma(query_input, top_k=top_k, filters=filters)
            else:
                if mode.startswith("By movie"):
                    hits = semantic_search_inmemory_title(query_input, top_k=top_k, filters=filters)
                else:
                    hits = semantic_search_inmemory_text(query_input, top_k=top_k, filters=filters)
        except Exception as e:
            st.error("حدث خطأ أثناء البحث: " + str(e))
            st.stop()

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
                            imdb_id = details.get('external_ids', {}).get('imdb_id') if details.get('external_ids') else None
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
                    st.write(f"**Score/Distance:** {round(h.get('score', 0), 6)}")
                    st.markdown("---")
                enhanced.append({
                    "title": h['title'],
                    "year": h.get('year'),
                    "genres": h.get('genres'),
                    "description": h.get('description'),
                    "score": h.get('score'),
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
    st.info("اكتب استعلام واضغط Search. App يحاول استخدام chroma إن متاح وإلا يعمل fallback باستخدام embeddings محلياً.")

# Footer: dataset info
with st.expander("Dataset info"):
    st.write(f"Rows: {len(df)}")
    st.write(f"Unique titles: {df['title'].nunique()}")
    st.write("Sample titles:")
    st.write(df['title'].drop_duplicates().head(50).tolist())
