import asyncio
import concurrent.futures
import functools
import sqlite3
import threading
import time

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .parser import parse_citation
from .normalization import normalize_title
from .fusion import fuse_candidates
from .live_lookup import live_lookup_verify
from .verifier import heuristic_verify, llm_verify, llm_verify_direct

app = FastAPI(title="Citation Validator API")


@app.on_event("startup")
def startup():
    from .semantic_search import load_model

    load_model()


class ValidateRequest(BaseModel):
    citation: str


class BatchValidateRequest(BaseModel):
    citations: list[str]


_s2_lock = threading.Lock()
_s2_last_call = 0.0


def _acquire_s2_slot():
    global _s2_last_call
    with _s2_lock:
        now = time.monotonic()
        wait = max(0, config.S2_RATE_LIMIT - (now - _s2_last_call))
        if wait > 0:
            time.sleep(wait)
        _s2_last_call = time.monotonic()


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


def _verify_single(raw_citation: str, *, batch_mode: bool = False) -> dict:
    raw = raw_citation.strip()
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
    result = heuristic_verify(top)
    source = "db_heuristic"

    if config.USE_LLM and fused:
        top_c = fused[0]
        f_sim = top_c.get("fuzzy_score", 0.0) / 100.0
        s_raw = top_c.get("semantic_score", -1.0)
        s_sim = 1.0 / (1.0 + s_raw) if s_raw >= 0 else 0.0
        match_q = (f_sim + s_sim) / 2.0

        if match_q >= 0.6:
            llm_result = llm_verify(fused, parsed)
            if llm_result:
                result = llm_result
                source = "llm_deepseek"
        elif result.get("label") == "PARTIALLY_VALID":
            llm_result = llm_verify(fused, parsed)
            if llm_result and llm_result.get("label") == "VALID":
                result = llm_result
                source = "llm_deepseek"

    if config.USE_LIVE_LOOKUP and result.get("label") == "HALLUCINATED":
        if live_lookup_cache:
            result = live_lookup_cache
            source = result.get("source", "live_lookup")
        else:
            if batch_mode:
                _acquire_s2_slot()
            live_result = live_lookup_verify(parsed)
            if live_result:
                result = live_result
                source = result.get("source", "live_lookup")

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


@app.post("/validate")
def validate(req: ValidateRequest) -> dict:
    return _verify_single(req.citation, batch_mode=False)


@app.post("/validate_batch")
async def validate_batch(req: BatchValidateRequest) -> dict:
    citations = [c.strip() for c in req.citations if c.strip()]
    if not citations:
        return {"results": []}
    if len(citations) > config.MAX_BATCH_SIZE:
        return {"error": f"Max {config.MAX_BATCH_SIZE} citations per batch"}

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(citations)) as pool:
        futures = [
            loop.run_in_executor(pool, functools.partial(_verify_single, batch_mode=True), c)
            for c in citations
        ]
        done_set, _ = await asyncio.wait(futures, timeout=config.BATCH_TIMEOUT)

    results = []
    for i, fut in enumerate(futures):
        if fut in done_set:
            try:
                r = fut.result()
                r.update({"index": i, "timed_out": False})
            except Exception as e:
                r = {
                    "index": i,
                    "label": "ERROR",
                    "confidence": 0.0,
                    "source": "",
                    "top_matches": [],
                    "reason": str(e),
                    "live_match": None,
                    "timed_out": False,
                }
        else:
            fut.cancel()
            r = {
                "index": i,
                "label": "TIMEOUT",
                "confidence": 0.0,
                "source": "",
                "top_matches": [],
                "reason": "Batch processing time exceeded",
                "live_match": None,
                "timed_out": True,
            }
        results.append(r)

    return {"results": sorted(results, key=lambda r: r["index"])}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
