import json
import re
import urllib.parse
from urllib.error import URLError
from urllib.request import Request, urlopen

from rapidfuzz import fuzz

from . import config
from .parser import ieee_author_overlap

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TIMEOUT = 15


def _search_papers(query: str, limit: int = 5) -> list[dict] | None:
    api_key = config.SEMANTIC_SCHOLAR_API_KEY
    url = "{}?query={}&limit={}&fields=title,authors,year,externalIds,venue,publicationVenue,abstract".format(
        SEARCH_URL, urllib.parse.quote(query), limit
    )
    req = Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
    req.add_header("User-Agent", "mailto:live-lookup@example.com")
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except (URLError, json.JSONDecodeError, OSError):
        return None


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


def _venue_similarity(p_venue: str, api_venue: str) -> float:
    if not p_venue or not api_venue:
        return 0.0
    stopwords = frozenset({
        "in", "on", "of", "the", "and", "for", "a", "an", "to", "with",
        "at", "by", "is", "was", "are", "its", "their", "this", "that",
        "from", "as", "be", "or", "but", "not", "so", "if", "it", "we",
        "our", "all", "no", "has", "have", "been", "were", "can", "will",
        "do", "did", "each", "any", "also", "than", "then",
    })
    def strip(text):
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(
            t for t in clean.split()
            if t not in stopwords
            and any(c.isalpha() for c in t)
        )
    clean_p, clean_a = strip(p_venue), strip(api_venue)
    tokens_p = set(clean_p.split())
    tokens_a = set(clean_a.split())
    if not tokens_p or not tokens_a:
        return 0.0
    overlap = len(tokens_p & tokens_a) / min(len(tokens_p), len(tokens_a))
    if overlap >= 0.2:
        return fuzz.token_set_ratio(clean_p, clean_a) / 100.0
    return 0.0


def _doi_similarity(api_doi: str | None, parsed_doi: str) -> float:
    if not api_doi or not parsed_doi:
        return 0.0
    a = api_doi.lower().strip()
    p = parsed_doi.lower().strip()
    if a == p:
        return 1.0
    if p in a or a in p:
        return 0.8
    return 0.0


TITLE_W = 0.18
AUTHOR_W = 0.25
YEAR_W = 0.11
VENUE_W = 0.05
DOI_W = 0.10
SEMANTIC_W = 0.31
API_SCORE_CAP_VALID = 0.90
API_SCORE_CAP_PARTIAL = 0.70
VALID_THRESHOLD = 0.55
PARTIAL_THRESHOLD = 0.35


def live_lookup_verify(parsed: dict) -> dict | None:
    query = parsed.get("title", "")
    if not query:
        return None

    results = _search_papers(query)
    if not results:
        return None

    p_authors = parsed.get("authors", "") or ""
    p_year = parsed.get("year")
    p_venue = parsed.get("venue", "") or ""
    p_doi = parsed.get("doi", "") or ""

    best_score = 0.0
    best_paper = None

    for paper in results:
        title = paper.get("title", "")
        authors_list = paper.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list if a.get("name"))
        year = paper.get("year")
        venue = paper.get("venue")
        if not venue:
            pv = paper.get("publicationVenue")
            if pv:
                venue = pv.get("name")
        venue = venue or ""

        ext_ids = paper.get("externalIds") or {}
        api_doi = ext_ids.get("DOI", "") or ""
        abstract = paper.get("abstract") or ""

        title_sim = _token_overlap(query.lower(), title.lower())
        author_sim = max(
            _token_overlap(p_authors.lower(), authors.lower()) if p_authors else 0.0,
            ieee_author_overlap(p_authors, authors) if p_authors else 0.0,
        )
        year_sim = _year_similarity(year, p_year)
        venue_sim = _venue_similarity(p_venue, venue)
        doi_sim = _doi_similarity(api_doi, p_doi)

        score = (
            TITLE_W * title_sim
            + AUTHOR_W * author_sim
            + YEAR_W * year_sim
            + VENUE_W * venue_sim
            + DOI_W * doi_sim
        )

        if score > best_score:
            best_score = score
            best_paper = {
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "doi": api_doi,
                "abstract": abstract,
            }

    if best_score >= VALID_THRESHOLD:
        return {
            "label": "VALID",
            "confidence": round(min(best_score, API_SCORE_CAP_VALID), 4),
            "reason": "Paper found via Semantic Scholar live lookup",
            "live_match": best_paper,
            "source": "live_lookup",
        }

    if best_score >= PARTIAL_THRESHOLD:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(min(best_score, API_SCORE_CAP_PARTIAL), 4),
            "reason": "Partial match found via Semantic Scholar live lookup",
            "live_match": best_paper,
            "source": "live_lookup",
        }

    return None
