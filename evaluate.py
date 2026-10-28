"""
Evaluate the trained intent classifier: prints accuracy, macro F1, and a
confusion matrix. Also saves a confusion matrix heatmap as a PNG.

Usage:
    python evaluate.py
"""
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from nlp_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "intents.json"
MODEL_PATH = BASE_DIR / "models" / "intent_pipeline.joblib"


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(intent["tag"])

    pipeline = joblib.load(MODEL_PATH)

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("=" * 55)
    print(f"  Intent classification accuracy : {acc*100:.1f}%")
    print(f"  Macro F1 score                  : {f1:.3f}")
    print(f"  Intents covered                 : {len(set(labels))}")
    print(f"  Total training phrases          : {len(texts)}")
    print("=" * 55)
    print()
    print(classification_report(y_test, preds, zero_division=0))

    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Intent Classification Confusion Matrix")
        for i in range(len(labels_sorted)):
            for j in range(len(labels_sorted)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        out_path = BASE_DIR / "models" / "confusion_matrix.png"
        fig.savefig(out_path, dpi=140)
        print(f"\nSaved confusion matrix heatmap to {out_path}")
    except ImportError:
        print("\n(matplotlib not installed — skipping confusion matrix PNG. "
              "Install matplotlib to generate it: pip install matplotlib)")
        print("\nConfusion matrix (rows=actual, cols=predicted):")
        print(" " * 16 + "  ".join(f"{l[:6]:>6s}" for l in labels_sorted))
        for i, row_label in enumerate(labels_sorted):
            print(f"{row_label:16s}" + "  ".join(f"{v:6d}" for v in cm[i]))


if __name__ == "__main__":
    main()
