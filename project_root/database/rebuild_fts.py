import os
import sqlite3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_SCRIPT_DIR, "papers.db")


def rebuild_fts():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-800000")

    conn.execute("DROP TABLE IF EXISTS papers_fts")

    schema_path = os.path.join(_SCRIPT_DIR, "schema.sql")
    with open(schema_path) as f:
        conn.executescript(f.read())

    total = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE title IS NOT NULL OR normalized_title IS NOT NULL"
    ).fetchone()[0]
    print(f"Rebuilding FTS index for {total} papers...")

    min_id, max_id = conn.execute(
        "SELECT MIN(paper_id), MAX(paper_id) FROM papers WHERE title IS NOT NULL OR normalized_title IS NOT NULL"
    ).fetchone()

    inserted = 0
    chunk = 50000

    for start in range(min_id, max_id + 1, chunk):
        end = start + chunk - 1
        rows = conn.execute(
            "SELECT paper_id, title, normalized_title, abstract "
            "FROM papers "
            "WHERE (title IS NOT NULL OR normalized_title IS NOT NULL) AND paper_id BETWEEN ? AND ?",
            (start, end),
        ).fetchall()

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO papers_fts(rowid, title, normalized_title, abstract) VALUES (?, ?, ?, ?)",
                rows,
            )
            inserted += len(rows)
            if inserted % 200000 == 0 or inserted == total:
                conn.commit()
                print(f"  Indexed {inserted:,} / {total:,}")

    conn.commit()
    conn.close()
    print(f"FTS index rebuilt: {inserted:,} rows indexed ({total - inserted} skipped).")


if __name__ == "__main__":
    rebuild_fts()
