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
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
LIMIT = 100
SLEEP_SECONDS = 1.0
MAX_OFFSET = 9999  # Semantic Scholar caps at ~10k results without API key

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_SCRIPT_DIR, "papers.db")


def normalize_title(title):
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_authors(authors_list):
    if not authors_list:
        return None
    names = [a.get("name") for a in authors_list if a.get("name")]
    return ", ".join(names) if names else None


def extract_venue(publication_venue):
    if publication_venue and publication_venue.get("name"):
        return publication_venue["name"]
    return None


def fetch_papers(search_query, offset=0, max_retries=3, max_429_retries=10):
    params = urllib.parse.urlencode({
        "query": search_query,
        "limit": LIMIT,
        "offset": offset,
        "fields": "title,authors,year,externalIds,abstract,publicationVenue",
    })
    url = f"{BASE_URL}?{params}"
    for attempt in range(1, max_429_retries + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "mailto:s2-ingest@example.com")
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
                wait = min(10 * (2 ** (attempt - 1)), 120)
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

    offset = 0
    inserted = 0
    batch = 0

    while offset <= MAX_OFFSET:
        data = fetch_papers(search_query, offset)
        results = data.get("data", [])
        if not results:
            break

        batch += 1
        batch_inserted = 0

        for paper in results:
            external_ids = paper.get("externalIds") or {}
            doi = external_ids.get("DOI")
            if not doi:
                continue

            doi_clean = doi.split("https://doi.org/")[-1]

            cur = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (doi_clean,))
            if cur.fetchone():
                continue

            title = paper.get("title")
            normalized_title = normalize_title(title) if title else None
            authors = extract_authors(paper.get("authors"))
            year = paper.get("year")
            venue = extract_venue(paper.get("publicationVenue"))
            abstract = paper.get("abstract")

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

        total = data.get("total", "?")
        next_offset = data.get("next")
        print(f"  Batch {batch}: {batch_inserted} inserted (total {inserted} so far, {total} available)")

        if max_records and inserted >= max_records:
            print(f"Reached target of {max_records} records.")
            break

        if next_offset is None:
            break

        offset = next_offset
        time.sleep(SLEEP_SECONDS)

    conn.close()
    print(f"Ingestion complete. {inserted} records inserted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest papers from Semantic Scholar API")
    parser.add_argument("query", help="Search query for the Semantic Scholar API")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of records to ingest (default: unlimited)")
    args = parser.parse_args()
    ingest_all(args.query, max_records=args.max_records)
