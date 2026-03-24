import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process

# Load
df = joblib.load("movie_recommender/df.pkl")
tfidf_matrix = joblib.load("movie_recommender/tfidf_matrix.pkl")

# Clean titles
df["title_clean"] = df["title"].str.lower().str.strip()

indices = pd.Series(df.index, index=df['title_clean']).drop_duplicates()

# 🔍 Smart search using fuzzy matching
def find_movie(user_input):
    titles = df["title"].tolist()
    match, score, _ = process.extractOne(user_input, titles)

    if score > 60:
        return match.lower().strip()
    return None

# 🎬 Recommendation function
def recommend(movie_title, top_n=5):
    matched = find_movie(movie_title)

    if matched is None or matched not in indices:
        return [{"title": "Movie not found", "score": 0}]

    idx = indices[matched]

    sim_scores = linear_kernel(
        tfidf_matrix[idx:idx+1],
        tfidf_matrix
    ).flatten()

    top_indices = np.argpartition(sim_scores, -(top_n+1))[-(top_n+1):]
    top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]

    results = []

    for i in top_indices:
        if i == idx:
            continue

        results.append({
            "title": df.iloc[i]["title"],
            "score": float(sim_scores[i])
        })

        if len(results) >= top_n:
            break

    return results