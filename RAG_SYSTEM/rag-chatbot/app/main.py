from fastapi import FastAPI
from .core.classifier import QueryClassifier

app = FastAPI(title="RAG Chatbot - Skeleton")

classifier = QueryClassifier()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(payload: dict):
    text = payload.get("text", "")
    return {"label": classifier.classify(text)}
