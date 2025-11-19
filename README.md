# Movie_Recommendation_System

This project is a semantic movie recommendation system built using:

- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS for fast similarity search
- Streamlit for the user interface
- TMDB API for fetching posters, ratings, and IMDb links

The system takes a movie title or a short description and returns the most semantically similar movies from a Netflix dataset.

Features

Semantic Search
The model understands the meaning of the text, not only keywords.

Vector Search using FAISS
All movies are converted to vector embeddings, and FAISS is used to retrieve similar movies efficiently.

TMDB Integration
For each recommended movie, the system fetches:
- Poster
- Rating
- IMDb link
