def heuristic_verify(top_candidate: dict) -> dict:
    if not top_candidate:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "No candidate provided",
        }

    fuzzy = top_candidate.get("fuzzy_score", 0.0)
    author_sim = top_candidate.get("author_similarity", 0.0)
    year_sim = top_candidate.get("year_similarity", 0.0)
    final = top_candidate.get("final_score", 0.0)

    title_sim = fuzzy / 100.0
    year_diff_ok = year_sim >= 0.5

    if title_sim > 0.90 and author_sim > 0.70 and year_diff_ok:
        conf = round(0.4 * title_sim + 0.3 * author_sim + 0.3 * year_sim, 4)
        return {
            "label": "VALID",
            "confidence": conf,
            "reason": "Title, author, and year all match within thresholds",
        }

    if (title_sim > 0.70 or final > 60) and (author_sim > 0.3 or year_diff_ok):
        conf = round(0.5 * title_sim + 0.3 * author_sim + 0.2 * year_sim, 4)
        return {
            "label": "PARTIALLY_VALID",
            "confidence": min(conf, 0.85),
            "reason": "Partial match found but some metadata is off",
        }

    return {
        "label": "HALLUCINATED",
        "confidence": round(max(0.0, title_sim * 0.3), 4),
        "reason": "Candidate does not meet minimum similarity thresholds",
    }


def test_heuristic_verify_high_similarity_returns_valid():
    candidate = {
        "fuzzy_score": 95.0,
        "author_similarity": 0.85,
        "year_similarity": 1.0,
        "final_score": 92.0,
    }
    result = heuristic_verify(candidate)
    assert result["label"] == "VALID"
    assert 0.0 < result["confidence"] <= 1.0
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0


def test_heuristic_verify_low_similarity_returns_hallucinated():
    candidate = {
        "fuzzy_score": 25.0,
        "author_similarity": 0.1,
        "year_similarity": 0.0,
        "final_score": 15.0,
    }
    result = heuristic_verify(candidate)
    assert result["label"] == "HALLUCINATED"
    assert 0.0 <= result["confidence"] < 0.5
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0


def test_heuristic_verify_empty_candidate_returns_hallucinated():
    result = heuristic_verify({})
    assert result["label"] == "HALLUCINATED"
    assert result["confidence"] == 0.0
    assert "No candidate" in result["reason"]


def test_heuristic_verify_borderline_partially_valid():
    candidate = {
        "fuzzy_score": 80.0,
        "author_similarity": 0.4,
        "year_similarity": 1.0,
        "final_score": 75.0,
    }
    result = heuristic_verify(candidate)
    assert result["label"] == "PARTIALLY_VALID"
    assert 0.0 < result["confidence"] <= 0.85
