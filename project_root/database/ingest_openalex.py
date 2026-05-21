import argparse
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
import http.client
import urllib.error


INCOMPLETE_READ_SLEEP = 5
MAX_429_RETRIES = 10


BASE_URL = "https://api.openalex.org/works"
PER_PAGE = 200
SLEEP_SECONDS = 1.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_SCRIPT_DIR, "papers.db")


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


def fetch_works(search_query, cursor="*", max_retries=3, max_429_retries=MAX_429_RETRIES):
    params = urllib.parse.urlencode({
        "search": search_query,
        "per_page": PER_PAGE,
        "cursor": cursor,
        "mailto": "openalex-ingest@example.com",
    })
    url = f"{BASE_URL}?{params}"
    for attempt in range(1, max_429_retries + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "mailto:openalex-ingest@example.com")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except json.JSONDecodeError:
            if attempt < max_retries:
                wait = 5 * (2 ** (attempt - 1))
                print(f"Invalid JSON response, retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(5 * (2 ** (attempt - 1)), 120)
                print(f"Rate limited, retrying in {wait}s (attempt {attempt}/{max_429_retries})...")
                time.sleep(wait)
                continue
            if 500 <= e.code <= 599 and attempt < max_429_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"Server error {e.code}, retrying in {wait}s "
                      f"(attempt {attempt}/{max_429_retries})...")
                time.sleep(wait)
                continue
            print(f"HTTP Error {e.code} for URL: {url}")
            raise
        except (http.client.IncompleteRead,
                urllib.error.URLError,
                ConnectionResetError,
                TimeoutError) as e:
            if attempt < max_retries:
                wait = INCOMPLETE_READ_SLEEP * (2 ** (attempt - 1))
                print(f"Connection error ({type(e).__name__}), retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"Connection error ({type(e).__name__}): {e}")
            raise


def ingest_all(search_query, max_records=None):
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    with open(schema_path, "r") as f:
        conn.executescript(f.read())

    cursor = "*"
    inserted = 0
    page = 0

    while True:
        data = fetch_works(search_query, cursor)
        results = data.get("results", [])
        if not results:
            break

        page += 1
        batch_inserted = 0

        for work in results:
            doi = work.get("doi")
            if not doi:
                continue

            doi_clean = doi.split("https://doi.org/")[-1]

            cur = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (doi_clean,))
            if cur.fetchone():
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
            batch_inserted += 1

            if max_records and inserted >= max_records:
                break

        conn.commit()

        total = data.get("meta", {}).get("count", "?")
        print(f"  Page {page}: {batch_inserted} inserted (total {inserted} so far, {total} available)")

        if max_records and inserted >= max_records:
            print(f"Reached target of {max_records} records.")
            break

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        cursor = next_cursor
        time.sleep(SLEEP_SECONDS)

    conn.close()
    print(f"Ingestion complete. {inserted} records inserted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest papers from OpenAlex API")
    parser.add_argument("query", help="Search query for the OpenAlex API")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of records to ingest (default: unlimited)")
    args = parser.parse_args()
    ingest_all(args.query, max_records=args.max_records)
