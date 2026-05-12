"""
FastAPI backend for Sentiment Analysis Dashboard
Run: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
from pathlib import Path  # ← make sure this line is here
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "Thinzar2003/sentiment-distilbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Fine-tuned DistilBERT for sentiment classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ─────────────────────────────────────────────────────
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    latency_ms: float

class BatchResult(BaseModel):
    results: List[SentimentResult]
    summary: dict

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(texts: List[str]) -> List[dict]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)

    id2label = model.config.id2label
    return [
        {"label": id2label[int(pred)], "score": float(probs[i][pred])}
        for i, pred in enumerate(preds)
    ]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running", "device": DEVICE}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=SentimentResult)
def predict_single(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.time()
    result = predict([input.text])[0]
    latency = round((time.time() - t0) * 1000, 2)

    return SentimentResult(
        text=input.text,
        label=result["label"],
        score=result["score"],
        latency_ms=latency,
    )

@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(input: BatchInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(input.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per request")

    t0 = time.time()
    raw = predict(input.texts)
    latency = round((time.time() - t0) * 1000, 2)

    results = [
        SentimentResult(
            text=text,
            label=r["label"],
            score=r["score"],
            latency_ms=latency / len(input.texts),
        )
        for text, r in zip(input.texts, raw)
    ]

    labels = [r.label for r in results]
    summary = {
        "total": len(results),
        "positive": labels.count("POSITIVE"),
        "negative": labels.count("NEGATIVE"),
        "avg_confidence": round(np.mean([r.score for r in results]), 4),
        "total_latency_ms": latency,
    }

    return BatchResult(results=results, summary=summary)