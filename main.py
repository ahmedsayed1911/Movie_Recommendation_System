import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

# ------- simplified semantic search (no filters, no TMDB) -------
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
        results = semantic_search(query, k=10)

        if not results:
            st.error("No results found")
        else:
            st.success(f"Found {len(results)} results")

            for idx, dist in results:
                row = df.iloc[idx]

                st.markdown(f"### {row['title']} ({row['year']})")
                st.write(f"Genres: {row['genres']}")
                st.write(row["description"][:400] + "...")
                st.write(f"Distance: {round(float(dist), 4)}")
                st.markdown("---")
