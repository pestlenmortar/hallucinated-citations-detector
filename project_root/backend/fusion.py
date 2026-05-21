import sqlite3


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _year_similarity(db_year, parsed_year) -> float:
    if db_year is None or parsed_year is None:
        return 0.0
    if db_year == parsed_year:
        return 1.0
    if abs(db_year - parsed_year) <= 1:
        return 0.5
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
        f"SELECT paper_id, title, authors, year, venue FROM papers WHERE paper_id IN ({placeholders})",
        list(paper_ids),
    ).fetchall()
    conn.close()

    db_info = {
        row[0]: {"title": row[1], "authors": row[2], "year": row[3], "venue": row[4]}
        for row in rows
    }

    p_authors = parsed_citation.get("authors", "") or ""
    p_year = parsed_citation.get("year")
    p_venue = parsed_citation.get("venue", "") or ""

    candidates = []
    for pid in paper_ids:
        fuzzy_entry = fuzzy_by_id.get(pid)
        semantic_entry = semantic_by_id.get(pid)

        fuzzy_score = fuzzy_entry["score"] if fuzzy_entry else 0.0
        semantic_score = semantic_entry["score"] if semantic_entry else 0.0

        info = db_info.get(pid, {})
        db_title = info.get("title") or ""
        db_authors = info.get("authors") or ""
        db_year = info.get("year")
        db_venue = info.get("venue") or ""

        author_sim = _token_overlap(p_authors, db_authors)
        year_sim = _year_similarity(db_year, p_year)
        venue_sim = _token_overlap(p_venue, db_venue)

        metadata_score = (author_sim + year_sim + venue_sim) / 3.0
        final_score = 0.5 * fuzzy_score + 0.4 * semantic_score + 0.1 * metadata_score

        candidates.append(
            {
                "paper_id": pid,
                "title": db_title,
                "authors": db_authors,
                "year": db_year,
                "venue": db_venue,
                "fuzzy_score": round(fuzzy_score, 4),
                "semantic_score": round(semantic_score, 4),
                "metadata_score": round(metadata_score, 4),
                "author_similarity": round(author_sim, 4),
                "year_similarity": round(year_sim, 4),
                "venue_similarity": round(venue_sim, 4),
                "final_score": round(final_score, 4),
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:5]
