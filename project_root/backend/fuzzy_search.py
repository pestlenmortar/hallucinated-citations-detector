import sqlite3

from rapidfuzz import fuzz, process


_title_cache = None


def _load_titles(db_path: str):
    global _title_cache
    if _title_cache is None:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT paper_id, title, normalized_title FROM papers WHERE normalized_title IS NOT NULL"
        )
        _title_cache = cursor.fetchall()
        conn.close()
    return _title_cache


def clear_title_cache():
    global _title_cache
    _title_cache = None


def fuzzy_search(query_title: str, db_path: str) -> list[dict]:
    rows = _load_titles(db_path)

    choices = [row[2] for row in rows]
    results = process.extract(
        query_title, choices, scorer=fuzz.token_sort_ratio, limit=10
    )

    output = []
    for _, score, idx in results:
        paper_id, title = rows[idx][0], rows[idx][1]
        output.append(
            {"paper_id": paper_id, "title": title, "score": round(score, 2)}
        )

    return output
