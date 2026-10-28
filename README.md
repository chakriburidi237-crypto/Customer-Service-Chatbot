# Customer Service AI Chatbot

> **"Intelligent Support. Instant Resolution. Powered by Modern NLP."**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)

Customer Service AI Chatbot is an end-to-end, production-grade conversational support system. It combines statistical NLP (TF-IDF vectorization with tuned multi-class Logistic Regression), rule-based named-entity extraction, multi-turn dialogue memory, confidence scoring with intelligent fallbacks, and a sleek real-time web portal. It resolves common e-commerce inquiries—such as order tracking, cancellation, refund requests, delivery windows, payment queries, and human escalation—in sub-millisecond latencies.

---

## 🚀 Key Features

1. **High-Accuracy Intent Classification**: Employs an optimized scikit-learn pipeline (sublinear TF-IDF + Logistic Regression with balanced class weighting) classifying incoming user queries into discrete customer service intent categories.
2. **Context-Aware Multi-Turn Conversations**: In-memory session tracking maintains state across user turns, remembering previously recognized entities (e.g. order numbers, dates, emails) so follow-up inquiries like *"Can I cancel it?"* resolve without re-asking.
3. **Automated Entity Extraction**: Rule-based regex tokenizers isolate order IDs (`#12345`), tracking numbers, email addresses, phone numbers, and amounts for contextual fulfillment.
4. **Confidence-Scored Fallback Mechanism**: If the highest predicted class probability drops below the configurable threshold ($\tau = 0.35$), the bot gracefully triggers a clarification dialog rather than outputting misleading information.
5. **Alternative Intent Suggestions**: Every chat response returns top alternative predictions with normalized probability distributions for explainability and operator telemetry.
6. **Built-In Model Training & Evaluation Suite**: Includes automated scripts (`train.py` and `evaluate.py`) that generate balanced confusion matrices, classification metrics (Precision, Recall, F1-Score), and serialize model artifacts to `models/`.
7. **Responsive Real-Time Chat Interface**: Modern web interface supporting message history, typing indicators, suggested quick-reply chips, and session resets.

---

## 🛠 Tech Stack

- **Framework**: FastAPI (ASGI with Uvicorn)
- **Machine Learning & NLP**: Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`), Joblib
- **Data & Evaluation**: Pandas, NumPy, Matplotlib (confusion matrix generation)
- **Language**: Python 3.10+
- **Frontend**: Clean Vanilla JavaScript (ES6+), HTML5, CSS3 with responsive chat bubbles and action buttons
- **Testing**: Python `unittest` suite covering multi-turn context retention, entity extraction, and fallback logic

---

## 🏗 Architecture & Directory Structure

```
Customer-Service-Chatbot/
├── data/
│   └── intents.json          # Annotated training corpus of user queries & intent tags
├── models/
│   ├── intent_pipeline.joblib# Serialized TF-IDF + Logistic Regression model
│   ├── responses.json        # Curated response templates per intent
│   └── confusion_matrix.png  # Evaluation performance heatmap
├── static/
│   └── index.html            # Web-based interactive chat interface
├── tests/
│   └── test_chatbot.py       # Automated regression test suite
├── app.py                    # FastAPI server, conversation session manager, and API
├── evaluate.py               # Model evaluation and confusion matrix generator
├── nlp_utils.py              # Text cleaning and regex entity extraction utilities
├── train.py                  # Training pipeline script
├── requirements.txt          # Python dependencies
├── .gitattributes            # Linguist language definitions
├── .gitignore                # Environment and cache rules
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## ⚡ Getting Started

### Prerequisites

- Python `3.10` or higher
- `pip` package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chakriburidi237-crypto/Customer-Service-Chatbot.git
   cd Customer-Service-Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Re-train or evaluate the model:
   ```bash
   python train.py
   python evaluate.py
   ```

5. Run the development server:
   ```bash
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## 🧪 Automated Testing

Automated unit tests assert correct intent classification, entity persistence across dialogue turns, fallback triggering on unanswerable messages, and session lifecycle:

```bash
python -m unittest discover tests -v
```

Expected output:
```
test_order_message_returns_intent_entity_and_rankings (test_chatbot.ChatbotBehaviorTests) ... ok
test_order_number_is_carried_into_follow_up (test_chatbot.ChatbotBehaviorTests) ... ok
test_session_reset_removes_context (test_chatbot.ChatbotBehaviorTests) ... ok
test_unrecognized_message_requests_clarification (test_chatbot.ChatbotBehaviorTests) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.082s

OK
```

---

## 🚀 Deployment

### Deploy to Render / Railway

1. Connect your repository to [Render](https://render.com).
2. Set Build Command: `pip install -r requirements.txt && python train.py`
3. Set Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📄 License and Copyright\n\nAll rights reserved © 2026 Chakradhar Buridi.

