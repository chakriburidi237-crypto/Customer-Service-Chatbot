"""
NLP preprocessing and lightweight entity extraction utilities used by both
train.py and app.py.
"""
import re

ORDER_NUMBER_PATTERN = re.compile(
    r"(?:#\s?\d{4,10})|(?:\bORD[- ]?[A-Z0-9]{4,10}\b)|(?:\border\s*(?:number|no\.?|#)?\s*[:#]?\s*(\d{4,10})\b)",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs and punctuation noise while keeping order-number
    style tokens (#12345, ORD-1234) intact for the classifier's own use is not
    needed here since entity extraction runs on the raw text separately."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s#\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(raw_text: str) -> dict:
    """Pull out order numbers and dates from the raw (unprocessed) message."""
    entities = {}

    order_match = ORDER_NUMBER_PATTERN.search(raw_text)
    if order_match:
        entities["order_number"] = order_match.group(0).strip()

    date_match = DATE_PATTERN.search(raw_text)
    if date_match:
        entities["date"] = date_match.group(0).strip()

    return entities
