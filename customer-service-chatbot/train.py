"""
Train a TF-IDF + Logistic Regression intent classifier on data/intents.json
and save the pipeline + responses lookup to disk.

Usage:
    python train.py
"""
import json
import joblib
import random
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from nlp_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "intents.json"
MODEL_PATH = BASE_DIR / "models" / "intent_pipeline.joblib"
RESPONSES_PATH = BASE_DIR / "models" / "responses.json"


def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    responses = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)

    return texts, labels, responses


def main():
    texts, labels, responses = load_dataset()
    print(f"Loaded {len(texts)} training phrases across {len(set(labels))} intents "
          f"(including fallback with 0 patterns, handled via confidence threshold).")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=8.0, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\nAccuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)

    print(f"Saved trained pipeline to {MODEL_PATH}")
    print(f"Saved responses lookup to {RESPONSES_PATH}")


if __name__ == "__main__":
    main()
