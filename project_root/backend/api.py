import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .parser import parse_citation
from .normalization import normalize_title
from .fusion import fuse_candidates
from .verifier import heuristic_verify, llm_verify

app = FastAPI(title="Citation Validator API")


class ValidateRequest(BaseModel):
    citation: str


def _exact_db_lookup(normalized_title: str, db_path: str) -> list[dict]:
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

    exact = _exact_db_lookup(normed, config.DB_PATH)
    fuzzy = _try_fuzzy(parsed.get("title") or normed, config.DB_PATH)
    sem = _try_semantic(parsed.get("title") or normed, config.FAISS_INDEX_PATH)

    all_fuzzy = exact + fuzzy
    fused = fuse_candidates(all_fuzzy, sem, parsed, config.DB_PATH)

    top = fused[0] if fused else {}
    result = heuristic_verify(top)

    if config.USE_LLM and fused:
        llm_result = llm_verify(fused, parsed)
        if llm_result.get("label"):
            result = llm_result

    return {
        "label": result.get("label", "HALLUCINATED"),
        "confidence": result.get("confidence", 0.0),
        "top_matches": fused,
        "reason": result.get("reason", ""),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
