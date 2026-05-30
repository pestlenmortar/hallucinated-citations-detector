"""
Insert ~30 missing well-known ML papers into the existing papers.db
by querying the OpenAlex API.
"""
import csv
import json
import re
import sqlite3
import time
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError


# ── API ──────────────────────────────────────────────────────────────────────
OPENALEX_BASE = "https://api.openalex.org/works"
USER_AGENT = "CitationValidator/1.0 (mailto:example@example.com)"
PAGE_SIZE = 10
API_DELAY = 0.6  # seconds between calls (polite to the free API)


def _openalex_request(url: str) -> dict | None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError) as e:
        print(f"    API error: {e}")
        return None


# ── NORMALIZATION (same as backend/normalization.py) ─────────────────────────
def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── FIND A PAPER ──────────────────────────────────────────────────────────────
def search_openalex_by_title(title: str) -> list[dict]:
    """Search OpenAlex by title. Returns list of work dicts."""
    query = title.replace('"', "").strip().rstrip(".,; ")
    url = f"{OPENALEX_BASE}?search={quote(query)}&per-page={PAGE_SIZE}"
    data = _openalex_request(url)
    if data is None:
        return []
    results = data.get("results", [])
    # Filter to exact-ish title match only
    matches = []
    for r in results:
        r_title = (r.get("title") or "").strip()
        if normalize_title(r_title) == normalize_title(title):
            matches.append(r)
    return matches


def get_work_by_doi(doi: str) -> dict | None:
    """Fetch a work by its DOI."""
    if not doi:
        return None
    doi_clean = doi.strip().lower()
    if not doi_clean.startswith("10."):
        return None
    url = f"{OPENALEX_BASE}/doi:{doi_clean}"
    return _openalex_request(url)


# ── EXTRACT FIELDS ────────────────────────────────────────────────────────────
def extract_fields(work: dict) -> dict:
    """Extract the fields we need from an OpenAlex work record."""
    title = (work.get("title") or "").strip()
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None
    year = work.get("publication_year")
    venue_obj = work.get("primary_location", {}) or {}
    venue_src = venue_obj.get("source") or {}
    venue = venue_src.get("display_name") or ""

    # Authors
    authorships = work.get("authorships") or []
    author_names = []
    for a in authorships:
        au = a.get("author") or {}
        name = au.get("display_name", "")
        if name:
            author_names.append(name)
    authors = "; ".join(author_names) if author_names else None

    # Abstract (inverted index from OpenAlex)
    abstract_inverted = work.get("abstract_inverted_index") or {}
    if abstract_inverted:
        word_positions = []
        for word, positions in abstract_inverted.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        abstract = " ".join(w for _, w in word_positions)
    else:
        abstract = None

    return {
        "title": title,
        "normalized_title": normalize_title(title),
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "abstract": abstract,
    }


# ── INSERT ────────────────────────────────────────────────────────────────────
def insert_paper(conn: sqlite3.Connection, fields: dict) -> bool:
    try:
        conn.execute(
            """INSERT OR IGNORE INTO papers
               (title, normalized_title, authors, year, venue, doi, abstract)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                fields["title"],
                fields["normalized_title"],
                fields["authors"],
                fields["year"],
                fields["venue"],
                fields["doi"],
                fields["abstract"],
            ),
        )
        conn.commit()
        return conn.total_changes > 0
    except sqlite3.Error as e:
        print(f"    DB error: {e}")
        return False


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    conn = sqlite3.connect("database/papers.db")

    # Read all unique titles from test dataset that are real papers
    reader = csv.DictReader(open("evaluation/datasets/test_citations.csv"))
    seen = set()
    papers_to_add = []

    for row in reader:
        label = row["true_label"]
        if label == "HALLUCINATED":
            continue  # fake papers — don't insert

        title_match = re.search(r'"(.+?)"', row["raw_citation"])
        title = title_match.group(1) if title_match else ""
        normed = normalize_title(title)
        if normed in seen or not normed:
            continue
        seen.add(normed)

        # Skip if already in DB
        exists = conn.execute(
            "SELECT paper_id FROM papers WHERE normalized_title = ?", (normed,)
        ).fetchone()
        if exists is not None:
            continue

        papers_to_add.append(
            {
                "citation_id": row["citation_id"],
                "title": title,
                "normed": normed,
                "doi_from_citation": _extract_doi(row["raw_citation"]),
            }
        )

    print(f"Unique real papers missing from DB: {len(papers_to_add)}")
    print()

    inserted = 0
    failed = 0

    for i, paper in enumerate(papers_to_add, 1):
        title = paper["title"]
        print(f"[{i}/{len(papers_to_add)}] {title[:70]}")

        work = None

        # Try DOI first
        if paper["doi_from_citation"]:
            print(f"    Trying DOI: {paper['doi_from_citation']}")
            work = get_work_by_doi(paper["doi_from_citation"])
            time.sleep(API_DELAY)

        # Fall back to title search
        if work is None:
            print(f"    Searching by title ...")
            matches = search_openalex_by_title(title)
            time.sleep(API_DELAY)
            if matches:
                work = matches[0]
                if len(matches) > 1:
                    print(f"    Found {len(matches)} exact title matches, using first")
            else:
                # Broader search without filtering
                query = title.replace('"', "").strip().rstrip(".,; ")[:80]
                url = f"{OPENALEX_BASE}?search={quote(query)}&per-page=5"
                data = _openalex_request(url)
                time.sleep(API_DELAY)
                if data:
                    results = data.get("results", [])
                    if results:
                        r = results[0]
                        r_title = (r.get("title") or "").strip()
                        print(f"    Closest match: \"{r_title[:60]}...\" — skipping (not exact)")
                        work = None

        if work is None:
            print(f"    FAILED — could not find on OpenAlex")
            failed += 1
            continue

        fields = extract_fields(work)
        if insert_paper(conn, fields):
            inserted += 1
            print(f"    INSERTED: \"{fields['title'][:60]}\"  pid=???  year={fields['year']}")
        else:
            print(f"    SKIPPED (exists or error)")

    conn.close()
    print(f"\nDone. Inserted: {inserted}, Failed/Skipped: {failed}")
    print("Run `python database/rebuild_fts.py` to rebuild the FTS index.")


def _extract_doi(text: str) -> str:
    m = re.search(r"doi\s*:\s*(10\.\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".,; ")
    return ""


if __name__ == "__main__":
    main()
