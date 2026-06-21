"""
Unit tests for the preprocessing module (src/preprocessing.py).

Run with:
    pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    clean_text,
    clean_text_noise,
    normalize_text,
    build_reverse_product_map,
    get_target_raw_products,
    PRODUCT_MAP,
)


def test_clean_text_noise_lowercases():
    assert "hello world" in clean_text_noise("HELLO World")


def test_clean_text_noise_removes_urls():
    result = clean_text_noise("Visit our site: http://example.com now")
    assert "http" not in result
    assert "example.com" not in result


def test_clean_text_noise_removes_phone_numbers():
    result = clean_text_noise("Call me at 555-1234 please")
    assert "555" not in result


def test_clean_text_noise_removes_html_tags():
    result = clean_text_noise("I hate the <br>high fees")
    assert "<br>" not in result
    assert "high fees" in result


def test_clean_text_noise_removes_redaction_placeholders():
    result = clean_text_noise("My account XXXX was charged on XX/XX/2024")
    assert "xxxx" not in result


def test_clean_text_noise_removes_special_characters():
    result = clean_text_noise("Fee of $50.00!! Why??")
    assert "$" not in result
    assert "!" not in result


def test_clean_text_noise_handles_empty_string():
    assert clean_text_noise("") == ""


def test_clean_text_noise_handles_none_gracefully():
    assert clean_text_noise(None) == ""


def test_clean_text_noise_handles_nan_like_float():
    assert clean_text_noise(float("nan")) == ""


def test_normalize_text_removes_stopwords():
    result = normalize_text("this is a complaint about the credit card")
    assert "is" not in result.split()
    assert "the" not in result.split()


def test_normalize_text_lemmatizes_verbs():
    result = normalize_text("the app keeps crashing")
    assert "crash" in result


def test_normalize_text_handles_empty_string():
    assert normalize_text("") == ""


def test_clean_text_full_pipeline():
    text = "I am writing to file a complaint! My account XXXX was crashing. Call 555-1234."
    result = clean_text(text)
    assert "xxxx" not in result
    assert "555" not in result
    assert "crash" in result  # lemmatized form of "crashing"


def test_build_reverse_product_map_covers_all_categories():
    reverse_map = build_reverse_product_map()
    assert set(reverse_map.values()) == set(PRODUCT_MAP.keys())


def test_get_target_raw_products_matches_product_map():
    raw_products = get_target_raw_products()
    expected_count = sum(len(v) for v in PRODUCT_MAP.values())
    assert len(raw_products) == expected_count
    assert "Credit card" in raw_products