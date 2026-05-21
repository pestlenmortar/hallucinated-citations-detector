import argparse
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request


BASE_URL = "https://api.openalex.org/works"
PER_PAGE = 200
DB_PATH = "papers.db"
SLEEP_SECONDS = 0.1


def normalize_title(title):
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_authors(authorships):
    if not authorships:
        return None
    names = []
    for entry in authorships:
        author = entry.get("author", {})
        name = author.get("display_name")
        if name:
            names.append(name)
    return ", ".join(names) if names else None


def extract_venue(primary_location):
    if primary_location and primary_location.get("source"):
        return primary_location["source"].get("display_name")
    return None


def fetch_works(search_query, page=1):
    params = urllib.parse.urlencode({
        "search": search_query,
        "page": page,
        "per_page": PER_PAGE,
    })
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "mailto:openalex-ingest@example.com")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ingest_all(search_query):
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    with open(schema_path, "r") as f:
        conn.executescript(f.read())

    page = 1
    inserted = 0

    while True:
        data = fetch_works(search_query, page)
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            doi = work.get("doi")
            if not doi:
                continue

            doi_clean = doi.split("https://doi.org/")[-1]

            cursor = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (doi_clean,))
            if cursor.fetchone():
                continue

            title = work.get("title")
            normalized_title = normalize_title(title) if title else None
            authors = extract_authors(work.get("authorships"))
            year = work.get("publication_year")
            venue = extract_venue(work.get("primary_location"))
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

            conn.execute(
                """INSERT INTO papers
                   (title, normalized_title, authors, year, venue, doi, abstract)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (title, normalized_title, authors, year, venue, doi_clean, abstract),
            )
            inserted += 1

        conn.commit()

        meta = data.get("meta", {})
        count = int(meta.get("count", 0))
        total_pages = (count + PER_PAGE - 1) // PER_PAGE if count else 1

        if page >= total_pages:
            break

        page += 1
        time.sleep(SLEEP_SECONDS)

    conn.close()
    print(f"Ingestion complete. {inserted} records inserted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest papers from OpenAlex API")
    parser.add_argument("query", help="Search query for the OpenAlex API")
    args = parser.parse_args()
    ingest_all(args.query)
