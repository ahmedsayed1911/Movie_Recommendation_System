import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import os

st.set_page_config(page_title="Movie Recommender + Posters", layout="wide")

# -----------------------------
# Config / TMDB
# -----------------------------
# You should set TMDB_API_KEY as an environment variable on Streamlit Cloud or locally.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")  # on Streamlit Cloud set as secret
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflixData.csv")
    # try to normalize common columns
    # keep title, genres, overview, and try to get year from potential columns
    if 'title' not in df.columns:
        raise ValueError("Data must contain a 'title' column.")
    for col in ['genres', 'overview', 'release_date', 'year', 'release_year']:
        if col not in df.columns:
            df[col] = pd.NA
    # try to extract year from release_date if exists
    def extract_year(row):
        for c in ['year', 'release_year']:
            if pd.notna(row.get(c)):
                try:
                    return int(row[c])
                except:
                    pass
        rd = row.get('release_date')
        if pd.notna(rd):
            try:
                return int(str(rd)[:4])
            except:
                return pd.NA
        return pd.NA

    df['year_clean'] = df.apply(extract_year, axis=1)
    df['genres'] = df['genres'].fillna("").astype(str)
    df['overview'] = df['overview'].fillna("").astype(str)
    df['combined'] = (df['genres'] + " " + df['overview']).str.strip()
    return df

df = load_data()

# -----------------------------
# Similarity matrix
# -----------------------------
@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    sim = cosine_similarity(tfidf_matrix)
    return sim

similarity = build_similarity(df)

# -----------------------------
# TMDB helper functions (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def tmdb_search(title):
    """
    Search TMDB for a movie by title. Returns first match dict or None.
    """
    if not TMDB_API_KEY:
        return None
    params = {"api_key": TMDB_API_KEY, "query": title, "include_adult": False}
    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        return results[0] if results else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_get_movie(movie_id):
    """
    Get TMDB movie details (vote_average, poster_path, external_ids)
    """
    if not TMDB_API_KEY or not movie_id:
        return None
    try:
        # Get main details
        r = requests.get(TMDB_MOVIE_URL.format(movie_id), params={"api_key": TMDB_API_KEY}, timeout=10)
        r.raise_for_status()
        details = r.json()
        # Get external IDs to possibly obtain imdb_id
        r2 = requests.get(TMDB_MOVIE_URL.format(movie_id) + "/external_ids", params={"api_key": TMDB_API_KEY}, timeout=10)
        r2.raise_for_status()
        ext = r2.json()
        details['external_ids'] = ext
        return details
    except Exception:
        return None

# -----------------------------
# Recommendation logic
# -----------------------------
def recommend_movies(title, top_k=10, genre_filter=None, year_min=None, year_max=None):
    title = title.lower().strip()
    if title not in df['title'].str.lower().values:
        return None
    idx = df[df['title'].str.lower() == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]  # exclude itself
    recs = []
    for i, score in scores:
        row = df.iloc[i]
        # apply filters
        if genre_filter:
            # genre_filter is list of selected genres (strings)
            if not any(g.strip().lower() in row['genres'].lower() for g in genre_filter):
                continue
        y = row.get('year_clean')
        if pd.notna(y):
            if year_min is not None and y < year_min:
                continue
            if year_max is not None and y > year_max:
                continue
        recs.append({
            "title": row['title'],
            "genres": row['genres'],
            "overview": row['overview'],
            "year": int(y) if pd.notna(y) else None,
            "sim_score": float(score)
        })
        if len(recs) >= top_k:
            break
    return recs

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommender (Posters · Ratings · Filters · Download)")

with st.sidebar:
    st.header("Options")
    movie_input = st.text_input("اكتب اسم الفيلم (exact or close):")
    top_k = st.slider("عدد التوصيات", min_value=5, max_value=20, value=10)
    # genres multiselect from data
    # split genres by common separators if needed
    all_genres = set()
    for g in df['genres'].dropna().astype(str):
        for part in g.split('|'):
            p = part.strip()
            if p:
                all_genres.add(p)
    genre_list = sorted(all_genres)
    selected_genres = st.multiselect("فلتر حسب النوع (Genres):", options=genre_list)
    # year slider - deduce min/max from data
    years = df['year_clean'].dropna().astype(int)
    if not years.empty:
        ymin = int(years.min())
        ymax = int(years.max())
    else:
        ymin, ymax = 1900, 2025
    year_range = st.slider("فلتر حسب السنة:", min_value=1900, max_value=2025, value=(ymin, ymax))
    use_tmdb = st.checkbox("اجلب Posters وRatings من TMDB (مطلوب TMDB_API_KEY)", value=True)
    st.markdown("---")
    st.markdown("**ملاحظات:**\n- ضع `TMDB_API_KEY` كـ secret في Streamlit Cloud أو كمتغير بيئي محليًا.\n- نستخدم تقييم TMDB (`vote_average`) كبديل رقمي لتقييم IMDb، ونوفّر رابط IMDb إن وُجد.")

# Main area: Recommend button
if st.button("Recommend"):
    if not movie_input or movie_input.strip() == "":
        st.warning("اكتب اسم الفيلم أولاً.")
    else:
        year_min, year_max = year_range
        recs = recommend_movies(movie_input, top_k=top_k, genre_filter=selected_genres, year_min=year_min, year_max=year_max)
        if recs is None:
            st.error("الفيلم غير موجود في الداتا.")
        elif len(recs) == 0:
            st.info("موجود الفيلم لكن فلترة النوع/السنة قضت على النتائج — جرّب إلغاء بعض الفلاتر.")
        else:
            # For each recommendation, optionally fetch TMDB details
            enhanced = []
            cols = st.columns(1)
            st.success(f"Found {len(recs)} recommendations:")
            for r in recs:
                title = r['title']
                poster_url = None
                rating = None
                imdb_link = None
                tmdb_id = None
                if use_tmdb and TMDB_API_KEY:
                    s = tmdb_search(title)
                    if s:
                        tmdb_id = s.get("id")
                        details = tmdb_get_movie(tmdb_id)
                        if details:
                            poster_path = details.get("poster_path")
                            if poster_path:
                                poster_url = TMDB_IMAGE_BASE + poster_path
                            rating = details.get("vote_average")  # TMDB rating
                            imdb_id = details.get("external_ids", {}).get("imdb_id")
                            if imdb_id:
                                imdb_link = f"https://www.imdb.com/title/{imdb_id}"
                # Build display card
                card_cols = st.columns([1, 3])
                with card_cols[0]:
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                    else:
                        st.write("No poster")
                with card_cols[1]:
                    st.markdown(f"### {title} ({r.get('year') or '—'})")
                    if rating:
                        st.write(f"**TMDB rating:** {rating} / 10")
                    if imdb_link:
                        st.write(f"[Open on IMDb]({imdb_link})")
                    st.write(f"**Genres:** {r.get('genres') or '—'}")
                    overview = r.get('overview') or ""
                    if overview:
                        st.write(overview[:500] + ("..." if len(overview) > 500 else ""))
                    st.write(f"**Similarity:** {round(r.get('sim_score', 0), 3)}")
                    st.markdown("---")
                # append to enhanced list
                enhanced.append({
                    "title": title,
                    "year": r.get('year'),
                    "genres": r.get('genres'),
                    "overview": r.get('overview'),
                    "similarity": r.get('sim_score'),
                    "tmdb_rating": rating,
                    "poster_url": poster_url,
                    "imdb_link": imdb_link
                })
            # Provide download button (CSV)
            df_out = pd.DataFrame(enhanced)
            csv_buffer = StringIO()
            df_out.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            st.download_button(
                label="Download recommendations (CSV)",
                data=csv_data,
                file_name=f"recommendations_for_{movie_input.replace(' ', '_')}.csv",
                mime="text/csv"
            )

# If the user didn't press recommend, show instructions or sample
else:
    st.info("اكتب اسم فيلم واضغط Recommend. يمكنك استخدام الفلاتر على اليسار لاختصار النتائج.")

