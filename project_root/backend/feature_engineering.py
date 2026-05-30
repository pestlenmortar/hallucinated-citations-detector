"""
Feature extraction from retrieval pipeline outputs.

Takes a top candidate dict (from fuse_candidates) plus parsed citation metadata
and produces a structured feature vector suitable for ML classifiers.

Improved version with richer features: interactions, nonlinear transforms,
character-level similarity, author order matching, DOI presence penalties.
"""

import re
from rapidfuzz import fuzz
import numpy as np

FEATURE_NAMES: list[str] = [
    "title_similarity",
    "author_overlap",
    "doi_match",
    "abstract_similarity",
    "semantic_similarity",
    "venue_similarity",
    "year_similarity",
    "fusion_final_score",
    "metadata_score",
    "exact_title_match",
    "exact_doi_match",
    "title_length_difference",
    "author_count_difference",
    "normalized_year_gap",
    "top_doi_match",
    # ── New features ──
    "title_char_similarity",
    "author_count_ratio",
    "title_word_count_diff_ratio",
    "title_author_interaction",
    "title_year_interaction",
    "semantic_title_interaction",
    "doi_presence_penalty",
    "first_author_match",
    "venue_similarity_sq",
    "author_overlap_sq",
    "fusion_metadata_ratio",
    "year_match_binary",
    "score_certainty",
    "semantic_boost",
    "title_final_interaction",
]

EXPECTED_DIM: int = len(FEATURE_NAMES)


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _first_author_match(parsed_authors: str, db_authors: str) -> int:
    if not parsed_authors or not db_authors:
        return 0
    pa = parsed_authors.replace(" and ", ", ").split(",")
    da = db_authors.replace(";", ",").split(",")
    if not pa or not da:
        return 0
    first_parsed = pa[0].strip().lower()
    first_db = da[0].strip().lower()
    if not first_parsed or not first_db:
        return 0
    if first_parsed == first_db:
        return 2
    pa_parts = set(first_parsed.split())
    db_parts = set(first_db.split())
    if pa_parts & db_parts:
        return 1
    return 0


def extract_feature_vector(
    top_candidate: dict,
    parsed_citation: dict,
    exact_title_match: int = 0,
    exact_doi_match: int = 0,
) -> np.ndarray:
    """
    Convert a retrieval candidate + parsed citation into a 1-D feature vector.

    Args:
        top_candidate: dict from fuse_candidates (must contain keys like
                       fuzzy_score, author_similarity, etc.)
        parsed_citation: dict from parse_citation (title, authors, year, doi)
        exact_title_match: 1 if normalized_title matched a DB record, else 0
        exact_doi_match: 1 if parsed DOI matches DB DOI for that record, else 0

    Returns:
        np.ndarray of shape (EXPECTED_DIM,), dtype float64.
    """
    top = top_candidate or {}
    parsed = parsed_citation or {}

    fuzzy_score: float = float(top.get("fuzzy_score", 0.0))
    title_similarity: float = fuzzy_score / 100.0

    author_overlap: float = float(top.get("author_similarity", 0.0))
    doi_sim: float = float(top.get("doi_similarity", 0.0))
    abstract_sim: float = float(top.get("abstract_similarity", 0.0))
    venue_sim: float = float(top.get("venue_similarity", 0.0))
    year_sim: float = float(top.get("year_similarity", 0.0))

    sem_score: float = float(top.get("semantic_score", -1.0))
    semantic_similarity: float = 1.0 / (1.0 + sem_score) if sem_score >= 0 else 0.0

    fusion_final: float = float(top.get("final_score", 0.0))
    metadata_score: float = float(top.get("metadata_score", 0.0))

    p_title: str = (parsed.get("title") or "").strip()
    p_doi: str = (parsed.get("doi") or "").strip().lower()
    p_authors: str = (parsed.get("authors") or "").strip()

    db_title: str = (top.get("title") or "").strip()
    db_doi: str = (top.get("doi") or "").strip().lower()
    db_authors: str = (top.get("authors") or "").strip()
    db_year = top.get("year")
    p_year = parsed.get("year")

    # ── Original engineered features ──
    p_len: int = len(p_title)
    d_len: int = len(db_title)
    max_len: int = max(p_len, d_len, 1)
    title_length_difference: float = abs(p_len - d_len) / float(max_len)

    p_auth_count: int = len(
        [a for a in p_authors.replace(" and ", ", ").split(",") if a.strip()]
    )
    d_auth_count: int = len(
        [a for a in db_authors.replace(";", ",").split(",") if a.strip()]
    )
    author_count_difference: float = float(abs(p_auth_count - d_auth_count))

    py_val: float = float(p_year or 0)
    dy_val: float = float(db_year or 0)
    max_yr: float = max(abs(py_val), abs(dy_val), 1.0)
    normalized_year_gap: float = (
        abs(py_val - dy_val) / max_yr if (p_year is not None and db_year is not None) else 1.0
    )

    top_doi_match: int = 1 if (p_doi and db_doi and p_doi == db_doi) else 0

    # ── New features ──

    # Character-level title similarity (complements token_sort_ratio)
    title_char_similarity: float = 0.0
    if p_title and db_title:
        title_char_similarity = fuzz.ratio(p_title.lower(), db_title.lower()) / 100.0

    # Author count ratio (more informative than raw difference)
    author_count_ratio: float = 0.0
    if p_auth_count > 0 or d_auth_count > 0:
        author_count_ratio = min(p_auth_count, d_auth_count) / max(p_auth_count, d_auth_count, 1)

    # Title word count difference ratio
    p_words = len(p_title.split())
    d_words = len(db_title.split())
    max_words = max(p_words, d_words, 1)
    title_word_count_diff_ratio: float = abs(p_words - d_words) / float(max_words)

    # Interaction features
    title_author_interaction: float = title_similarity * author_overlap
    title_year_interaction: float = title_similarity * year_sim
    semantic_title_interaction: float = semantic_similarity * title_similarity
    title_final_interaction: float = title_similarity * fusion_final / 100.0

    # DOI presence penalty: if DB has DOI but parsed citation doesn't
    doi_presence_penalty: float = 1.0 if (db_doi and not p_doi) else 0.0

    # First author match
    first_author_match: float = float(_first_author_match(p_authors, db_authors))

    # Nonlinear transforms
    venue_similarity_sq: float = venue_sim ** 2
    author_overlap_sq: float = author_overlap ** 2

    # Fusion-to-metadata ratio (captures how much fusion relies on metadata vs content)
    fusion_metadata_ratio: float = 0.0
    if metadata_score > 0 and fusion_final > 0:
        fusion_metadata_ratio = metadata_score * 100 / max(fusion_final, 0.01)

    # Year match binary (more stable than continuous year gap)
    year_match_binary: float = 1.0 if year_sim >= 1.0 else 0.0

    # Score certainty: difference between best and second-best score signals
    score_certainty: float = 0.0
    if top and top.get("final_score", 0) > 30:
        score_certainty = min(title_similarity, author_overlap, max(semantic_similarity, 0.1))

    # Semantic boost: how much semantic sim exceeds token-based similarity
    semantic_boost: float = max(0.0, semantic_similarity - title_similarity)

    features: np.ndarray = np.array(
        [
            title_similarity,
            author_overlap,
            doi_sim,
            abstract_sim,
            semantic_similarity,
            venue_sim,
            year_sim,
            fusion_final,
            metadata_score,
            float(exact_title_match),
            float(exact_doi_match),
            title_length_difference,
            author_count_difference,
            normalized_year_gap,
            float(top_doi_match),
            title_char_similarity,
            author_count_ratio,
            title_word_count_diff_ratio,
            title_author_interaction,
            title_year_interaction,
            semantic_title_interaction,
            doi_presence_penalty,
            first_author_match,
            venue_similarity_sq,
            author_overlap_sq,
            fusion_metadata_ratio,
            year_match_binary,
            score_certainty,
            semantic_boost,
            title_final_interaction,
        ],
        dtype=np.float64,
    )

    return features


def features_from_row(row: dict) -> np.ndarray:
    """Rebuild a feature vector from a CSV row dict (used for dataset loading)."""
    values = []
    for name in FEATURE_NAMES:
        values.append(float(row.get(name, 0.0)))
    return np.array(values, dtype=np.float64)
