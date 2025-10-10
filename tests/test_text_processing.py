import pytest

from src.utils import text_processing as tp


def test_clean_text_removes_noise() -> None:
    text = " Hello, WORLD! Call 1234. "
    cleaned = tp.clean_text(
        text,
        remove_punctuation=True,
        lowercase=True,
        remove_numbers=True,
    )
    assert cleaned == "hello world call"


def test_extract_keywords_ignores_stopwords() -> None:
    text = "Quickly the agent quickly resolved the issue and ensured quick follow-up."
    keywords = tp.extract_keywords(text, top_k=3, min_length=4)
    assert "agent" in keywords
    assert "quickly" in keywords
    assert "the" not in keywords


@pytest.mark.parametrize("method", ["jaccard", "cosine", "levenshtein"])
def test_calculate_similarity_identical_texts(method: str) -> None:
    text = "call analytics system"
    score = tp.calculate_similarity(text, text, method=method)
    assert score == pytest.approx(1.0)


def test_calculate_similarity_invalid_method() -> None:
    with pytest.raises(ValueError):
        tp.calculate_similarity("call", "call", method="unsupported")


def test_mask_pii_replaces_sensitive_tokens() -> None:
    text = "Contact me at user@example.com or 555-123-4567; card 4111 1111 1111 1111."
    masked = tp.mask_pii(text)
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "[CREDIT_CARD]" in masked
