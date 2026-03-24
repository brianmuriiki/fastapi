from fastapi import FastAPI
from pydantic import BaseModel
from movie_recommender.recommender import recommend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 CORS MIDDLEWARE (fixes 405 / OPTIONS error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development (allow all)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class MovieRequest(BaseModel):
    title: str

# Home route
@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}

# Recommendation route
@app.post("/recommend")
def get_recommendations(request: MovieRequest):
    results = recommend(request.title)
    return {"recommendations": results}