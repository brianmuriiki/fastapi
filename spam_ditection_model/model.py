import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("large_spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔍 Show class balance
print("Class distribution:\n", df['label'].value_counts())

# Split data (IMPORTANT: stratify to avoid imbalance issues)
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']   # 🔥 VERY IMPORTANT
)

# 🚀 Strong, stable pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),     # captures phrases
        max_df=0.9             # ignore overly common words
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"   # 🔥 handles imbalance
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 🧪 Manual sanity test
print("\nManual Tests:")
print("Hello friend →", model.predict(["Hello friend"])[0])
print("Win money now!!! →", model.predict(["Win money now!!!"])[0])

# Save model
joblib.dump(model, "spam_model.pkl")

print("\n✅ Model saved as spam_model.pkl")