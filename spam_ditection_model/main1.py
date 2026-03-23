from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("spam_model.pkl")

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Spam Detection API running 🚀"}

@app.post("/predict")
def predict(data: Message):
    prediction = model.predict([data.message])[0]
    probability = model.predict_proba([data.message])[0][1]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(probability * 100, 2)
    }