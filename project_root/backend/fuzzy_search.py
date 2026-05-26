import re
import sqlite3

from rapidfuzz import fuzz, process


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def _build_fts_query(query_title: str) -> str | None:
    tokens = _tokenize(query_title)
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return None
    return " AND ".join(f"{t}*" for t in tokens)


def fuzzy_search(query_title: str, db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)

    fts_query = _build_fts_query(query_title)
    if fts_query:
        try:
            cursor = conn.execute(
                "SELECT rowid, title, normalized_title FROM papers_fts WHERE papers_fts MATCH ?",
                (fts_query,),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                "SELECT paper_id, title, normalized_title FROM papers WHERE normalized_title IS NOT NULL"
            ).fetchall()
    else:
        rows = []

    if not rows:
        conn.close()
        return []

    choices = [row[2] for row in rows]
    results = process.extract(
        query_title, choices, scorer=fuzz.token_sort_ratio, limit=25
    )

    output = []
    for _, score, idx in results:
        paper_id, title = rows[idx][0], rows[idx][1]
        output.append(
            {"paper_id": paper_id, "title": title, "score": round(score, 2)}
        )

    conn.close()
    return output
