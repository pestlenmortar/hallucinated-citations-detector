import sqlite3

from rapidfuzz import fuzz, process


def fuzzy_search(query_title: str, db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT paper_id, title, normalized_title FROM papers WHERE normalized_title IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

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
