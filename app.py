"""
FastAPI backend for the Customer Service AI Chatbot.

Run:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""
import json
import random
from time import perf_counter
from pathlib import Path
from collections import defaultdict

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nlp_utils import clean_text, extract_entities

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "intent_pipeline.joblib"
RESPONSES_PATH = BASE_DIR / "models" / "responses.json"

CONFIDENCE_THRESHOLD = 0.35

app = FastAPI(
    title="Customer Service AI Chatbot API",
    description="NLP intent-classification chatbot (TF-IDF + Logistic Regression) "
                 "with confidence-based fallback, regex entity extraction, and "
                 "session-based multi-turn context.",
    version="1.0.0",
)

_pipeline = None
_responses = None

# In-memory per-session conversation state: session_id -> {"last_intent": str, "entities": dict}
_sessions: dict = defaultdict(lambda: {"last_intent": None, "entities": {}})


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run `python train.py` first.")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_responses():
    global _responses
    if _responses is None:
        with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
            _responses = json.load(f)
    return _responses


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class IntentAlternative(BaseModel):
    intent: str
    confidence: float


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    entities: dict
    alternatives: list[IntentAlternative]
    needs_clarification: bool
    processing_ms: float


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="message field cannot be empty")

    started_at = perf_counter()
    pipeline = get_pipeline()
    responses = get_responses()
    session = _sessions[payload.session_id]

    cleaned = clean_text(payload.message)
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best_idx = proba.argmax()
    confidence = float(proba[best_idx])
    predicted_intent = classes[best_idx]
    alternatives = [
        {"intent": classes[index], "confidence": round(float(proba[index]), 4)}
        for index in proba.argsort()[::-1][:3]
    ]

    # Route to fallback if the model isn't confident enough
    intent = predicted_intent if confidence >= CONFIDENCE_THRESHOLD else "fallback"

    # Extract entities from this turn's raw message
    entities = extract_entities(payload.message)

    # Multi-turn context: if this turn didn't mention an order number but the
    # previous turn did (and this turn continues the same intent-family),
    # reuse the entity from session state.
    if "order_number" not in entities and session["entities"].get("order_number"):
        if intent in {"order_status", "refund_request", "cancel_order", "complaint", "payment_issue"}:
            entities["order_number"] = session["entities"]["order_number"]
            entities["order_number_source"] = "carried over from earlier in this conversation"

    # Update session state
    session["last_intent"] = intent
    session["entities"].update({k: v for k, v in entities.items() if k != "order_number_source"})

    reply_template = random.choice(responses.get(intent, responses["fallback"]))

    # Personalize reply slightly if we have an order number and the intent needs it
    if entities.get("order_number") and intent in {"order_status", "refund_request", "cancel_order"}:
        reply = f"{reply_template} (Order reference: {entities['order_number']})"
    else:
        reply = reply_template

    return ChatResponse(
        reply=reply,
        intent=intent,
        confidence=round(confidence, 4),
        entities={k: v for k, v in entities.items() if k != "order_number_source"},
        alternatives=alternatives,
        needs_clarification=intent == "fallback",
        processing_ms=round((perf_counter() - started_at) * 1000, 2),
    )


@app.get("/intents")
def list_intents():
    responses = get_responses()
    return {"intents": list(responses.keys())}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TF-IDF + Logistic Regression",
        "intents": len(get_pipeline().classes_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared"}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
