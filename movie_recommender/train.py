import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Load dataset
df = pd.read_csv("movie_recommender/movies_dataset.csv")

# Clean
df["genre"] = df["genre"].fillna("").str.lower()
df["keywords"] = df["keywords"].fillna("").str.lower()

# Weighted features
df["features"] = df["genre"] + " " + df["genre"] + " " + df["keywords"]

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=2,
    max_df=0.8,
    max_features=8000
)

tfidf_matrix = tfidf.fit_transform(df["features"])
tfidf_matrix = normalize(tfidf_matrix)

# Save
joblib.dump(df, "df.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")

print("🔥 Model trained")