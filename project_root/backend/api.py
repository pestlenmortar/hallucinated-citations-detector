import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .parser import parse_citation, ieee_author_overlap
from .normalization import normalize_title
from .fusion import fuse_candidates, _token_overlap, _year_similarity
from .live_lookup import live_lookup_verify
from .verifier import heuristic_verify, llm_verify, llm_verify_direct, verify_top_candidate

app = FastAPI(title="Citation Validator API")


@app.on_event("startup")
def startup():
    from .semantic_search import load_model

    load_model()


class ValidateRequest(BaseModel):
    citation: str


AUTHOR_MATCH_THRESHOLD = 0.80


def _exact_lookup(parsed: dict, normed: str, db_path: str) -> list[dict] | None:
    """Return top_matches list if strict exact match passes, None otherwise.

    Strict match: title must be in DB AND (DOI matches OR authors>=0.80 + year exact).
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT paper_id, title, authors, year, doi FROM papers WHERE normalized_title = ?",
            (normed,),
        ).fetchone()
        if row is None:
            return None

        paper_id, db_title, db_authors, db_year, db_doi = row
        p_doi = (parsed.get("doi") or "").strip().lower()
        p_authors = (parsed.get("authors") or "").strip()
        p_year = parsed.get("year")

        if p_doi and db_doi:
            if db_doi.strip().lower() == p_doi:
                auth_overlap = max(
                    _token_overlap(p_authors, db_authors),
                    ieee_author_overlap(p_authors, db_authors),
                ) if p_authors and db_authors else 0.0
                year_ok = p_year is not None and db_year is not None and _year_similarity(db_year, p_year) == 1.0
                if auth_overlap >= 0.60 and year_ok:
                    return [{"paper_id": paper_id, "title": db_title, "score": 100.0}]
                return None  # DOI matches but metadata discrepancy — let pipeline decide
            return None

        if not p_doi:
            if p_authors and db_authors and p_year is not None and db_year is not None:
                if _year_similarity(db_year, p_year) == 1.0:
                    overlap = max(
                        _token_overlap(p_authors, db_authors),
                        ieee_author_overlap(p_authors, db_authors),
                    )
                    if overlap >= AUTHOR_MATCH_THRESHOLD:
                        return [{"paper_id": paper_id, "title": db_title, "score": 100.0}]
            return None
        return None
    except sqlite3.Error:
        return None
    finally:
        try:
            conn.close()
        except NameError:
            pass


def _exact_db_lookup(normalized_title: str, db_path: str) -> list[dict]:
    """Simple title-only exact match -- returns candidates for fusion."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT paper_id, title FROM papers WHERE normalized_title = ?",
            (normalized_title,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return [{"paper_id": row[0], "title": row[1], "score": 100.0}]
    except sqlite3.Error:
        pass
    return []


def _try_fuzzy(query: str, db_path: str) -> list[dict]:
    try:
        from .fuzzy_search import fuzzy_search

        return fuzzy_search(query, db_path)
    except ImportError:
        return []


def _try_semantic(query: str, index_path: str) -> list[dict]:
    try:
        from .semantic_search import semantic_search

        return semantic_search(query, index_path)
    except (ImportError, FileNotFoundError, OSError):
        return []


@app.post("/validate")
def validate(req: ValidateRequest) -> dict:
    raw = req.citation.strip()
    if not raw:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "top_matches": [],
            "reason": "Empty citation provided",
        }

    parsed = parse_citation(raw).model_dump()
    normed = normalize_title(parsed.get("title") or "")

    # Strict exact match: title+DOI or (no DOI, authors>=0.80 + exact year)
    strict_match = _exact_lookup(parsed, normed, config.DB_PATH)
    if strict_match is not None:
        return {
            "label": "VALID",
            "confidence": 1.0,
            "source": "db_exact",
            "top_matches": strict_match,
            "reason": "Strict exact match (title+DOI or authors+year)",
        }

    # Broad exact match: title-only (used as candidate, not a gate)
    exact = _exact_db_lookup(normed, config.DB_PATH)
    fuzzy = _try_fuzzy(parsed.get("title") or normed, config.DB_PATH)

    live_semantic_query = None
    live_lookup_cache = None
    if config.USE_LIVE_LOOKUP:
        live_result = live_lookup_verify(parsed)
        if live_result:
            live_lookup_cache = live_result
            abstract = (live_result.get("live_match") or {}).get("abstract") or ""
            if len(abstract) > len(parsed.get("title") or ""):
                live_semantic_query = abstract

    semantic_query = live_semantic_query or (parsed.get("title") or normed)
    sem = _try_semantic(semantic_query, config.FAISS_INDEX_PATH)

    all_fuzzy = exact + fuzzy
    fused = fuse_candidates(all_fuzzy, sem, parsed, config.DB_PATH)

    top = fused[0] if fused else {}
    exact_title_match = 1 if exact else 0
    result = verify_top_candidate(top, parsed, exact_title_match=exact_title_match)
    source = result.get("source", "db_classifier")

    # LLM verification (if enabled and candidate quality is good)
    if config.USE_LLM and fused:
        top_c = fused[0]
        f_sim = top_c.get("fuzzy_score", 0.0) / 100.0
        s_raw = top_c.get("semantic_score", -1.0)
        s_sim = 1.0 / (1.0 + s_raw) if s_raw >= 0 else 0.0
        match_q = (f_sim + s_sim) / 2.0

        if match_q >= 0.72:
            llm_result = llm_verify(fused, parsed)
            if llm_result:
                result = llm_result
                source = "llm_deepseek"
        elif result.get("label") == "PARTIALLY_VALID" and match_q >= 0.50:
            llm_result = llm_verify(fused, parsed)
            if llm_result and llm_result.get("label") == "VALID" and llm_result.get("confidence", 0.0) >= 0.85:
                result = llm_result
                source = "llm_deepseek"

    # Live lookup integration: only override if the classifier is confident
    # something exists, NOT when it says HALLUCINATED with low confidence
    if config.USE_LIVE_LOOKUP:
        result_label = result.get("label", "HALLUCINATED")
        result_conf = result.get("confidence", 0.0)

        if result_label == "HALLUCINATED":
            if result_conf >= 0.55:
                if live_lookup_cache:
                    result = live_lookup_cache
                    source = result.get("source", "live_lookup")
                else:
                    live_result = live_lookup_verify(parsed)
                    if live_result:
                        result = live_result
                        source = result.get("source", "live_lookup")
        else:
            if live_lookup_cache and live_lookup_cache.get("label") == "VALID":
                ll_score = live_lookup_cache.get("confidence", 0.0)
                if ll_score >= 0.80:
                    live_lookup_cache["source"] = "live_lookup"
                    return {
                        k: v for k, v in {
                            "label": live_lookup_cache.get("label"),
                            "confidence": live_lookup_cache.get("confidence", 0.0),
                            "source": "live_lookup",
                            "top_matches": fused,
                            "reason": live_lookup_cache.get("reason", ""),
                            "live_match": live_lookup_cache.get("live_match"),
                        }.items() if v is not None
                    }

    # Direct LLM fallback for HALLUCINATED (only when LLM is configured)
    if config.USE_LLM and result.get("label") == "HALLUCINATED":
        llm_result = llm_verify_direct(parsed)
        if llm_result:
            result = llm_result
            source = "llm_deepseek"

    return {
        k: v for k, v in {
            "label": result.get("label", "HALLUCINATED"),
            "confidence": result.get("confidence", 0.0),
            "source": source,
            "top_matches": fused,
            "reason": result.get("reason", ""),
            "live_match": result.get("live_match"),
        }.items() if v is not None
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
