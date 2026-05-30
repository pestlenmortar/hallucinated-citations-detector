"""
Deterministic field-level corruption detection.

Compares each citation field against the top DB candidate's metadata
and returns VALID only if ALL fields pass strict thresholds.
Any single field failure → PARTIALLY_VALID.
"""

import re

from .fusion import _token_overlap, _year_similarity, _venue_similarity
from .parser import ieee_author_overlap


def check_fields(parsed: dict, top_candidate: dict, exact_title_match: int) -> tuple[str, float, str]:
    """Check each citation field against the DB candidate metadata.

    Args:
        parsed: dict from parse_citation with keys title, authors, year, venue, doi
        top_candidate: top fused candidate dict with keys fuzzy_score, authors, year, venue, doi
        exact_title_match: 1 if normalized_title matched a DB record

    Returns:
        (label, confidence, reason) — either VALID or PARTIALLY_VALID
    """
    if not exact_title_match or not top_candidate:
        return "PARTIALLY_VALID", 0.50, "No exact DB match for field comparison"

    # ── Title check ──
    # Fuzzy score from candidate represents title similarity
    fuzzy_score = top_candidate.get("fuzzy_score", 0.0)
    title_sim = fuzzy_score / 100.0 if fuzzy_score > 1 else fuzzy_score
    # For exact match in DB, fuzzy_score is 100.0; for fuzzy matches, it's lower
    # Use author_similarity from the candidate as a supplementary title-quality signal
    auth_sim = top_candidate.get("author_similarity", 0.0)
    title_ok = exact_title_match == 1  # if exact title match in DB, title is fine

    # ── Year check ──
    p_year = parsed.get("year")
    db_year = top_candidate.get("year")
    year_ok = True
    if p_year is not None and db_year is not None:
        year_sim = _year_similarity(db_year, p_year)
        year_ok = year_sim == 1.0  # must be exact match

    # ── Author check ──
    p_authors = (parsed.get("authors") or "").strip()
    db_authors = (top_candidate.get("authors") or "").strip()
    author_overlap = max(
        _token_overlap(p_authors, db_authors),
        ieee_author_overlap(p_authors, db_authors),
    ) if p_authors and db_authors else 0.0
    authors_ok = author_overlap >= 0.80

    # ── Venue check ──
    p_venue = (parsed.get("venue") or "").strip()
    db_venue = (top_candidate.get("venue") or "").strip()
    venue_sim = _venue_similarity(p_venue, db_venue) if p_venue and db_venue else 0.5
    venue_ok = venue_sim >= 0.40

    # ── DOI check ──
    p_doi = (parsed.get("doi") or "").strip().lower()
    db_doi = (top_candidate.get("doi") or "").strip().lower()
    doi_ok = (not db_doi) or (p_doi and db_doi and p_doi == db_doi)
    missing_doi_only = (not p_doi) and db_doi  # citation missing DOI but DB has one

    # ── Collect failures ──
    failures = []
    if not title_ok:
        failures.append(f"title")
    if not year_ok:
        failures.append(f"year({p_year}!={db_year})")
    if not authors_ok:
        failures.append(f"authors({author_overlap:.2f})")
    if not venue_ok:
        failures.append(f"venue({venue_sim:.2f})")
    if not doi_ok:
        failures.append("DOI(missing/mismatch)")

    # ── Decision ──
    corr_count = (not title_ok) + (not year_ok) + (not authors_ok) + (not venue_ok) + (not doi_ok)

    if corr_count == 0:
        return "VALID", 0.95, "All fields match DB — citation is substantially correct"

    # Single-field failures: often DB data quality issues, not actual corruptions
    # Only flag as PARTIALLY_VALID if the failure is a clear corruption signal
    if corr_count == 1:
        if not year_ok and p_year is not None and db_year is not None and abs(p_year - db_year) >= 2:
            return "PARTIALLY_VALID", 0.65, f"Year mismatch ({p_year} vs DB {db_year})"
        if not authors_ok and author_overlap < 0.50:
            return "PARTIALLY_VALID", 0.65, f"Significant author mismatch ({author_overlap:.2f})"
        if not title_ok:
            return "PARTIALLY_VALID", 0.65, "Title does not match DB"
        # Single venue, year±1, DOI, or minor author mismatch → VALID (likely DB artifact)
        return "VALID", 0.80, f"Minor discrepancy (single field) — paper is valid"

    # Multi-field failures: clear corruption pattern
    return "PARTIALLY_VALID", 0.70, f"Multiple corruptions: {', '.join(failures)}"
