import sqlite3
import re

from rapidfuzz import fuzz

VENUE_STOPWORDS = frozenset({
    "in", "on", "of", "the", "and", "for", "a", "an", "to", "with",
    "at", "by", "is", "was", "are", "its", "their", "this", "that",
    "from", "as", "be", "or", "but", "not", "so", "if", "it", "we",
    "our", "all", "no", "has", "have", "been", "were", "can", "will",
    "do", "did", "each", "any", "also", "than", "then",
})

RANK_TITLE_W = 0.18
RANK_AUTHOR_W = 0.21
RANK_YEAR_W = 0.11
RANK_VENUE_W = 0.05
RANK_DOI_W = 0.10
RANK_SEMANTIC_W = 0.35


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _venue_similarity(p_venue: str, db_venue: str) -> float:
    if not p_venue or not db_venue:
        return 0.0
    def strip(text):
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(
            t for t in clean.split()
            if t not in VENUE_STOPWORDS
            and any(c.isalpha() for c in t)
        )
    clean_p, clean_d = strip(p_venue), strip(db_venue)
    tokens_p = set(clean_p.split())
    tokens_d = set(clean_d.split())
    if not tokens_p or not tokens_d:
        return 0.0
    overlap = len(tokens_p & tokens_d) / min(len(tokens_p), len(tokens_d))
    if overlap >= 0.2:
        return fuzz.token_set_ratio(clean_p, clean_d) / 100.0
    return 0.0


def _year_similarity(db_year, parsed_year) -> float:
    if db_year is None or parsed_year is None:
        return 0.0
    if db_year == parsed_year:
        return 1.0
    if abs(db_year - parsed_year) <= 1:
        return 0.5
    return 0.0


def _doi_similarity(db_doi: str | None, parsed_doi: str) -> float:
    if not db_doi or not parsed_doi:
        return 0.0
    db_clean = db_doi.lower().strip()
    p_clean = parsed_doi.lower().strip()
    if db_clean == p_clean:
        return 1.0
    if p_clean in db_clean or db_clean in p_clean:
        return 0.8
    return 0.0


def fuse_candidates(
    fuzzy_results: list,
    semantic_results: list,
    parsed_citation: dict,
    db_path: str,
) -> list[dict]:
    fuzzy_by_id = {r["paper_id"]: r for r in fuzzy_results if r.get("paper_id")}
    semantic_by_id = {r["paper_id"]: r for r in semantic_results if r.get("paper_id")}

    paper_ids = set(fuzzy_by_id) | set(semantic_by_id)
    if not paper_ids:
        return []

    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT paper_id, title, authors, year, venue, doi FROM papers WHERE paper_id IN ({placeholders})",
        list(paper_ids),
    ).fetchall()
    conn.close()

    db_info = {
        row[0]: {"title": row[1], "authors": row[2], "year": row[3],
                  "venue": row[4], "doi": row[5]}
        for row in rows
    }

    p_authors = parsed_citation.get("authors", "") or ""
    p_year = parsed_citation.get("year")
    p_venue = parsed_citation.get("venue", "") or ""
    p_doi = parsed_citation.get("doi", "") or ""

    candidates = []
    for pid in paper_ids:
        fuzzy_entry = fuzzy_by_id.get(pid)
        semantic_entry = semantic_by_id.get(pid)

        fuzzy_score = fuzzy_entry["score"] if fuzzy_entry else 0.0
        semantic_score = round(semantic_entry["score"], 4) if semantic_entry else -1.0

        info = db_info.get(pid, {})
        db_title = info.get("title") or ""
        db_authors = info.get("authors") or ""
        db_year = info.get("year")
        db_venue = info.get("venue") or ""
        db_doi = info.get("doi") or ""

        title_sim = fuzzy_score / 100.0
        sem_sim = 1.0 / (1.0 + semantic_score) if semantic_entry is not None else 0.0
        author_sim = _token_overlap(p_authors, db_authors)
        year_sim = _year_similarity(db_year, p_year)
        venue_sim = _venue_similarity(p_venue, db_venue)
        doi_sim = _doi_similarity(db_doi, p_doi)

        metadata_score = (author_sim + year_sim + venue_sim + doi_sim) / 4.0
        final_score = round(
            RANK_TITLE_W * title_sim * 100
            + RANK_AUTHOR_W * author_sim * 100
            + RANK_YEAR_W * year_sim * 100
            + RANK_VENUE_W * venue_sim * 100
            + RANK_DOI_W * doi_sim * 100
            + RANK_SEMANTIC_W * sem_sim * 100,
            4,
        )

        candidates.append(
            {
                "paper_id": pid,
                "title": db_title,
                "authors": db_authors,
                "year": db_year,
                "venue": db_venue,
                "doi": db_doi,
                "fuzzy_score": round(fuzzy_score, 4),
                "semantic_score": round(semantic_score, 4),
                "metadata_score": round(metadata_score, 4),
                "author_similarity": round(author_sim, 4),
                "year_similarity": round(year_sim, 4),
                "venue_similarity": round(venue_sim, 4),
                "doi_similarity": round(doi_sim, 4),
                "final_score": final_score,
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:5]
