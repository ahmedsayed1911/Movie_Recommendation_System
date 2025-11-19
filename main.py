import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer
from io import BytesIO, StringIO

# ============ CONFIG ============
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Movie Recommender (FAISS)", layout="wide")


# ============ LOAD DATA ============
@st.cache_data
def load_data():
    df = pd.read_csv("netflixData.csv")
    df["title"] = df["Title"].astype(str)
    df["description"] = df["Description"].fillna("").astype(str)
    df["genres"] = df["Genres"].fillna("").astype(str)

    def extract_year(v):
        try:
            return int(str(v)[:4])
        except:
            return None

    df["year"] = df["Release Date"].apply(extract_year)
    df["combined"] = df["title"] + " " + df["genres"] + " " + df["description"]
    return df.reset_index(drop=True)


df = load_data()


# ============ EMBEDDINGS + FAISS ============
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


model = load_model()


@st.cache_resource
def build_faiss():
    embeddings = model.encode(
        df["combined"].tolist(), show_progress_bar=True, convert_to_numpy=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


index, embeddings = build_faiss()


# ============ TMDB HELPERS ============
@st.cache_data
def tmdb_search(title: str):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            TMDB_SEARCH_URL,
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=8,
        )
        js = r.json()
        return js.get("results", [None])[0]
    except:
        return None


@st.cache_data
def tmdb_details(movie_id):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            TMDB_MOVIE_URL.format(movie_id),
            params={"api_key": TMDB_API_KEY},
            timeout=8,
        )
        d = r.json()

        r2 = requests.get(
            TMDB_MOVIE_URL.format(movie_id) + "/external_ids",
            params={"api_key": TMDB_API_KEY},
            timeout=8,
        )
        d["external_ids"] = r2.json()
        return d
    except:
        return None


# ============ SEMANTIC SEARCH ============
def semantic_search(query, k=10, genre_filters=None, year_min=None, year_max=None):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k * 3)
    distances = distances[0]
    idxs = idxs[0]

    results = []
    for d, i in zip(distances, idxs):
        row = df.iloc[i]

        # genre filter
        if genre_filters:
            ok = False
            for g in genre_filters:
                if g.lower() in row["genres"].lower():
                    ok = True
            if not ok:
                continue

        # year filter
        if row["year"] is not None:
            if year_min and row["year"] < year_min:
                continue
            if year_max and row["year"] > year_max:
                continue

        results.append((i, float(d)))

        if len(results) >= k:
            break

    return results


# ============ UI ============
st.title("🎬 Movie Recommender — Semantic Search + FAISS + TMDB")

query = st.text_input("اكتب اسم الفيلم أو وصف…")
top_k = st.slider("عدد النتائج", 5, 30, 10)

# genre list
genres_all = sorted(
    set(sum([g.split(",") for g in df["genres"].fillna("").tolist()], []))
)
selected_genres = st.multiselect("فلتر حسب النوع", genres_all)

year_min, year_max = st.slider(
    "فلتر سنة الإصدار", min_value=1900, max_value=2025, value=(1900, 2025)
)

use_tmdb = st.checkbox("عرض Poster + Rating من TMDB", value=True)


# ============ SEARCH BUTTON ============
if st.button("Search"):
    if query.strip() == "":
        st.warning("اكتب اسم فيلم أو query!")
    else:
        results = semantic_search(
            query,
            k=top_k,
            genre_filters=selected_genres,
            year_min=year_min,
            year_max=year_max,
        )

        if not results:
            st.error("لا توجد نتائج")
        else:
            st.success(f"تم العثور على {len(results)} نتيجة")

            for idx, dist in results:
                row = df.iloc[idx]

                poster = None
                rating = None
                imdb_link = None

                if use_tmdb:
                    js = tmdb_search(row["title"])
                    if js:
                        details = tmdb_details(js["id"])
                        if details:
                            if details.get("poster_path"):
                                poster = TMDB_IMAGE_BASE + details["poster_path"]
                            rating = details.get("vote_average")
                            imdb_id = details["external_ids"].get("imdb_id")
                            if imdb_id:
                                imdb_link = f"https://www.imdb.com/title/{imdb_id}"

                c1, c2 = st.columns([1, 3])

                with c1:
                    if poster:
                        st.image(poster, use_column_width=True)
                    else:
                        st.write("No Poster")

                with c2:
                    st.markdown(f"### {row['title']} ({row['year']})")
                    if rating:
                        st.write(f"⭐ TMDB Rating: {rating} / 10")
                    if imdb_link:
                        st.write(f"[فتح على IMDb]({imdb_link})")

                    st.write(f"**Genres:** {row['genres']}")
                    st.write(row["description"][:400] + "...")

                    st.write(f"**Distance:** {round(dist, 4)}")

                st.markdown("---")
