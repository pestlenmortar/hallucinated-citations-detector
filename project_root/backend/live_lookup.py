import json
import urllib.parse
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import config

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TIMEOUT = 15


def _search_papers(query: str, limit: int = 5) -> list[dict] | None:
    api_key = config.SEMANTIC_SCHOLAR_API_KEY
    url = "{}?query={}&limit={}&fields=title,authors,year,externalIds,venue,publicationVenue".format(
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


def live_lookup_verify(parsed: dict) -> dict | None:
    query = parsed.get("title", "")
    if not query:
        return None

    results = _search_papers(query)
    if not results:
        return None

    p_authors = parsed.get("authors", "") or ""
    p_year = parsed.get("year")

    best_score = 0.0
    best_paper = None

    for paper in results:
        title = paper.get("title", "")
        authors_list = paper.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list if a.get("name"))
        year = paper.get("year")

        title_sim = _token_overlap(query.lower(), title.lower())
        author_sim = _token_overlap(p_authors.lower(), authors.lower()) if p_authors else 0.0
        year_sim = _year_similarity(year, p_year)

        score = 0.5 * title_sim + 0.3 * author_sim + 0.2 * year_sim

        if score > best_score:
            best_score = score
            venue = paper.get("venue")
            if not venue:
                pv = paper.get("publicationVenue")
                if pv:
                    venue = pv.get("name")
            best_paper = {
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
            }

    if best_score >= 0.7:
        return {
            "label": "VALID",
            "confidence": round(best_score, 4),
            "reason": "Paper found via Semantic Scholar live lookup",
            "live_match": best_paper,
        }

    if best_score >= 0.4:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(best_score, 4),
            "reason": "Partial match found via Semantic Scholar live lookup",
            "live_match": best_paper,
        }

    return None
