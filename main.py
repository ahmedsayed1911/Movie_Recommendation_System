import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Movie Recommender", layout="wide")

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

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

@st.cache_resource
def build_faiss():
    embeddings = model.encode(
        df["combined"].tolist(),
        show_progress_bar=False,
        convert_to_numpy=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss()


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


def semantic_search(query, k=10):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k)
    distances = distances[0]
    idxs = idxs[0]
    return list(zip(idxs, distances))


st.title("Movie Recommender")

query = st.text_input("Enter movie title or description")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a movie name or query!")
    else:
        results = semantic_search(query, k=20)

        if not results:
            st.error("No results found")
        else:
            st.success(f"Found {len(results)} results")

            for idx, dist in results:
                row = df.iloc[idx]

                poster = None
                rating = None
                imdb_link = None

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
                        st.write(f"⭐ TMDB Rating: **{rating} / 10**")

                    if imdb_link:
                        st.markdown(f"[IMDb Link]({imdb_link})")

                    st.write(f"Genres: {row['genres']}")
                    st.write(row["description"][:400] + "...")

                    st.write(f"Distance: {round(float(dist), 4)}")

                st.markdown("---")
