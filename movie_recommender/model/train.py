import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("../data/movies.csv")

movies["features"] = movies["genre"] + " " + movies["overview"]

tfidf = TfidfVectorizer(stop_words="english")

matrix = tfidf.fit_transform(movies["features"])

similarity = cosine_similarity(matrix)

pickle.dump(similarity, open("similarity.pkl", "wb"))
pickle.dump(movies, open("movies.pkl", "wb"))

print("Model trained successfully")
