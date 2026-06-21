"""
src/preprocessing.py

Text cleaning and product-category mapping utilities for the CFPB complaint
dataset, used in Task 1 (EDA & Preprocessing) and reused by Task 2 (chunking).

Cleaning approach follows the course's Applied EDA demo:
  1. clean_text_noise()  -> lowercase, strip URLs/phone numbers/IDs/HTML/punctuation
  2. normalize_text()    -> tokenize, remove stopwords, lemmatize (verbs + nouns)
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK corpora (no-ops if already present).
for _resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Maps our 4 target product categories to the raw CFPB `Product` column
# values that should be considered part of that category. CFPB has renamed
# these labels over the years, so multiple raw values can map to one category.
PRODUCT_MAP = {
    "Credit Card": [
        "Credit card",
        "Credit card or prepaid card",
    ],
    "Personal Loan": [
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Consumer Loan",
        "Payday loan",
    ],
    "Savings Account": [
        "Bank account or service",
        "Checking or savings account",
    ],
    "Money Transfer": [
        "Money transfers",
        "Money transfer, virtual currency, or money service",
    ],
}


def clean_text_noise(text) -> str:
    """
    Step 1 cleaning: lowercase + strip noise (URLs, phone numbers, simple
    IDs/IBANs, HTML tags, punctuation/special characters).

    Returns an empty string for None / NaN-like / empty input.
    """
    if text is None:
        return ""
    text = str(text)
    if text.strip() == "" or text.lower() == "nan":
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove phone numbers (7-digit and 10-digit formats)
    text = re.sub(r"\b\d{3}[-.\s]?\d{4}\b", "", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", text)

    # Remove simple IDs/IBANs (2 letters followed by 10+ digits)
    text = re.sub(r"\b[a-z]{2}\d{10,}\b", "", text, flags=re.I)

    # Remove CFPB redaction placeholders, e.g. "xxxx" / "xx/xx/xxxx"
    text = re.sub(r"x{2,}", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters (keep word chars + spaces)
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Step 2 cleaning: tokenize, remove English stopwords, and lemmatize
    (both as verbs and nouns) so that inflected forms collapse to a
    common root (e.g. "crashing"/"crashed" -> "crash").

    Expects already-noise-cleaned text (see clean_text_noise). Returns
    an empty string for empty input.
    """
    if not text:
        return ""

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    lemmas = [_LEMMATIZER.lemmatize(t, pos="v") for t in tokens]
    lemmas = [_LEMMATIZER.lemmatize(t, pos="n") for t in lemmas]

    return " ".join(lemmas)


def clean_text(text) -> str:
    """
    Full cleaning pipeline: clean_text_noise() followed by normalize_text().
    This is the single entry point used by the EDA notebook and Task 2.
    """
    noise_cleaned = clean_text_noise(text)
    return normalize_text(noise_cleaned)


def build_reverse_product_map(product_map: dict = PRODUCT_MAP) -> dict:
    """
    Flatten PRODUCT_MAP into {raw_product_label: clean_category_name}
    for use with df["Product"].map(...).
    """
    return {raw: clean for clean, raws in product_map.items() for raw in raws}


def get_target_raw_products(product_map: dict = PRODUCT_MAP) -> list:
    """Flat list of all raw Product values we want to keep."""
    return [p for plist in product_map.values() for p in plist]