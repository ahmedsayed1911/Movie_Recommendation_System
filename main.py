import os
import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from io import StringIO

# -----------------------------------
# Streamlit page
# -----------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# -----------------------------------
# TMDB API
# -----------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

def tmdb_get(title):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=10
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        movie = results[0]

        details = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie['id']}",
            params={"api_key": TMDB_API_KEY},
            timeout=10
        ).json()

        poster = movie.get("poster_path")
        poster_url = TMDB_IMAGE_BASE + poster if poster else None

        imdb_id = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie['id']}/external_ids",
            params={"api_key": TMDB_API_KEY}
        ).json().get("imdb_id")

        imdb_link = f"https://www.imdb.com/title/{imdb_id}" if imdb_id else None

        return {
            "poster": poster_url,
            "rating": details.get("vote_average"),
            "imdb_link": imdb_link
        }
    except:
        return None


# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflixData.csv")

df = load_data()

df["title"] = df["Title"].astype(str)
df["desc"] = df["Description"].fillna("").astype(str)
df["genres"] = df["Genres"].fillna("").astype(str)
df["year"] = df["Release Date"].fillna(0).astype(int)

# -----------------------------------
# Embedding model
# -----------------------------------
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()

# -----------------------------------
# Chroma (in-memory ONLY)
# -----------------------------------
@st.cache_resource
def init_chroma():
    client = chromadb.Client()  # in-memory
    collection = client.create_collection("movies")

    documents = (df["title"] + ". " + df["desc"]).tolist()
    ids = df.index.astype(str).tolist()
    embeddings = model.encode(documents).tolist()

    metas = []
    for _, row in df.iterrows():
        metas.append({
            "title": row["title"],
            "genres": row["genres"],
            "year": int(row["year"]) if row["year"] else None,
            "desc": row["desc"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metas
    )

    return collection

collection = init_chroma()

# -----------------------------------
# UI
# -----------------------------------
st.title("🎬 Movie Recommender — Semantic Search + Posters")

query = st.text_input("اكتب اسم فيلم أو وصف:")
topk = st.slider("عدد النتائج:", 5, 30, 10)

if st.button("Search"):
    if not query.strip():
        st.warning("اكتب اسم فيلم أو وصف")
    else:
        query_emb = model.encode([query]).tolist()

        res = collection.query(
            query_embeddings=query_emb,
            n_results=topk
        )

        hits = []
        for meta, distance in zip(res["metadatas"][0], res["distances"][0]):
            hits.append({
                "title": meta["title"],
                "genres": meta["genres"],
                "desc": meta["desc"],
                "year": meta["year"],
                "distance": distance
            })

        for h in hits:
            tmdb = tmdb_get(h["title"])

            col1, col2 = st.columns([1, 3])

            with col1:
                if tmdb and tmdb["poster"]:
                    st.image(tmdb["poster"], use_column_width=True)
                else:
                    st.write("no poster")

            with col2:
                st.markdown(f"### {h['title']} ({h['year']})")
                st.write(f"**Genres:** {h['genres']}")
                st.write(h["desc"][:500] + "...")
                st.write(f"Distance: {round(h['distance'],4)}")
                if tmdb and tmdb["rating"]:
                    st.write(f"TMDB Rating: {tmdb['rating']}")
                if tmdb and tmdb["imdb_link"]:
                    st.write(f"[IMDb]({tmdb['imdb_link']})")

        # download CSV
        out = pd.DataFrame(hits)
        st.download_button(
            "📥 Download as CSV",
            out.to_csv(index=False),
            "results.csv",
            "text/csv"
        )
