import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import os

st.set_page_config(page_title="Netflix Movie Recommender", layout="wide")

# -----------------------------
# TMDB SETTINGS
# -----------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflixData.csv")
    
    # Rename to consistent names
    df = df.rename(columns={
        "Title": "title",
        "Description": "overview",
        "Genres": "genres",
        "Imdb Score": "imdb",
        "Release Date": "year"
    })
    
    # Clean
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    
    df["combined"] = df["genres"] + " " + df["overview"]
    
    # extract numeric year
    def extract_year(x):
        try:
            return int(str(x).split(".")[0])
        except:
            return None
    
    df["year_clean"] = df["year"].apply(extract_year)
    return df

df = load_data()

# -----------------------------
# Similarity Model
# -----------------------------
@st.cache_resource
def build_similarity():
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    sim = cosine_similarity(matrix)
    return sim

similarity = build_similarity()

# -----------------------------
# TMDB FUNCTIONS
# -----------------------------
@st.cache_data
def tmdb_search(title):
    if not TMDB_API_KEY:
        return None

    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        r = requests.get(TMDB_SEARCH, params=params)
        data = r.json()
        if len(data.get("results", [])) > 0:
            return data["results"][0]
    except:
        return None
    
    return None

# -----------------------------
# Recommend
# -----------------------------
def recommend(movie, top_k=10, genre_filter=None, year_range=None):
    movie = movie.lower().strip()
    
    if movie not in df["title"].str.lower().values:
        return None

    idx = df[df["title"].str.lower() == movie].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

    out = []

    for i, s in scores:
        row = df.iloc[i]

        # Genre filter
        if genre_filter:
            if not any(g.lower() in row["genres"].lower() for g in genre_filter):
                continue
        
        # Year filter
        y = row["year_clean"]
        if year_range and y:
            if y < year_range[0] or y > year_range[1]:
                continue

        out.append({
            "title": row["title"],
            "genres": row["genres"],
            "overview": row["overview"],
            "year": y,
            "similarity": float(s),
            "imdb": row["imdb"]
        })

        if len(out) >= top_k:
            break
    
    return out

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Netflix Movie Recommendation System (Posters + IMDb + Filters)")

with st.sidebar:
    st.header("Filters")
    movie_name = st.text_input("اكتب اسم فيلم من الداتا:")
    top_k = st.slider("عدد التوصيات", 5, 20, 10)
    
    # genre filter
    all_genres = sorted(set(sum([g.split(",") for g in df["genres"].dropna()], [])))
    genre_filter = st.multiselect("فلتر حسب النوع:", all_genres)
    
    # year filter
    years = df["year_clean"].dropna().astype(int)
    if not years.empty:
        year_min = int(years.min())
        year_max = int(years.max())
    else:
        year_min, year_max = (1950, 2025)

    year_range = st.slider("فلتر حسب السنة:", year_min, year_max, (year_min, year_max))

    st.markdown("⚠ **ضع TMDB_API_KEY في Secrets لأجل الصور.**")

if st.button("Recommend"):
    if not movie_name:
        st.warning("اكتب اسم فيلم أولاً.")
    else:
        result = recommend(movie_name, top_k, genre_filter, year_range)

        if not result:
            st.error("لا توجد نتائج — جرب إزالة بعض الفلاتر.")
        else:
            st.success(f"تم العثور على {len(result)} فيلم مشابه:")

            final_list = []

            for r in result:
                # TMDB Poster
                poster = None
                tmdb_data = tmdb_search(r["title"])
                if tmdb_data and tmdb_data.get("poster_path"):
                    poster = TMDB_IMG + tmdb_data["poster_path"]

                col1, col2 = st.columns([1, 3])

                with col1:
                    if poster:
                        st.image(poster)
                    else:
                        st.write("No Poster")

                with col2:
                    st.markdown(f"### {r['title']} ({r['year']})")
                    st.write(f"⭐ IMDb: {r['imdb']}")
                    st.write(f"🎭 Genres: {r['genres']}")
                    st.write(r["overview"][:400] + "...")
                    st.write(f"🔗 Similarity: {round(r['similarity'], 3)}")

                st.markdown("---")

                final_list.append(r)

            # Download CSV
            df_out = pd.DataFrame(final_list)
            st.download_button(
                "⬇ تحميل النتائج CSV",
                df_out.to_csv(index=False),
                file_name=f"recommendations_{movie_name}.csv",
                mime="text/csv"
            )

else:
    st.info("اكتب اسم فيلم ثم اضغط Recommend 👍")
