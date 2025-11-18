import os
import streamlit as st
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# ---------------------------------------------------
# STREAMLIT SETTINGS
# ---------------------------------------------------
st.set_page_config(page_title="Movie Recommender (Semantic + Chroma)", layout="wide")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflixData.csv")

    # Normalize title column
    if "Title" not in df.columns:
        raise ValueError("Your dataset MUST contain a column named 'Title'.")

    df["title"] = df["Title"].astype(str)
    df["description"] = df["Description"].fillna("").astype(str)
    df["genres"] = df["Genres"].fillna("").astype(str)

    # Extract year
    def get_year(x):
        try:
            return int(str(x)[:4])
        except:
            return None

    if "Release Date" in df.columns:
        df["year"] = df["Release Date"].apply(get_year)
    else:
        df["year"] = None

    df["combined"] = df["title"] + " " + df["genres"] + " " + df["description"]
    return df

df = load_data()

# ---------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------------------------------
# CHROMA INIT
# ---------------------------------------------------
@st.cache_resource
def init_chroma():
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_store"
    )
    return chromadb.Client(settings)

client = init_chroma()

# Create / Load collection
COLLECTION_NAME = "movies_embeddings"
try:
    collection = client.get_collection(COLLECTION_NAME)
except:
    collection = client.create_collection(COLLECTION_NAME)

# ---------------------------------------------------
# POPULATE CHROMA IF EMPTY
# ---------------------------------------------------
def populate_chroma():
    if collection.count() > 0:
        return

    texts = df["combined"].tolist()
    ids = df.index.astype(str).tolist()

    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=df[["title", "genres", "description", "year"]].to_dict(orient="records"),
        embeddings=embeddings
    )

    client.persist()

populate_chroma()

# ---------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------
def semantic_search(query, top_k=10):
    emb = embedder.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=emb,
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    hits = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "title": meta.get("title", ""),
            "genres": meta.get("genres", ""),
            "description": meta.get("description", ""),
            "year": meta.get("year", None),
            "distance": dist
        })

    return hits

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🎬 Movie Recommender — Semantic Search (Chroma)")

query = st.text_input("اكتب اسم الفيلم أو وصف للفيلم:")

top_k = st.slider("عدد النتائج:", 5, 30, 10)

if st.button("Search"):
    if query.strip() == "":
        st.warning("من فضلك اكتب استعلام.")
    else:
        results = semantic_search(query, top_k)

        st.success(f"تم العثور على {len(results)} نتيجة")

        for r in results:
            st.markdown(f"""
                ### 🎥 {r['title']} ({r['year']})
                **Genres:** {r['genres']}  
                **Distance:** {round(r['distance'], 4)}  

                {r['description'][:400]}...
                ---
            """)

# Dataset Debug Info
with st.expander("Dataset Preview"):
    st.write(df.head())
