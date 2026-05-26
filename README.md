# Citation Validator — Multi-Citation Branch

This branch (`multi_citation`) extends the single-citation pipeline with batch processing, concurrent verification, in-memory caching, and Semantic Scholar rate-limit enforcement. It maintains full backward compatibility with the single-citation workflow on `master`.

---

## Branching: `master` vs `multi_citation`

| Aspect | `master` | `multi_citation` |
|---|---|---|
| Citations per request | 1 | 1 **or** many (up to 50) |
| Endpoint | `POST /validate` | `POST /validate` + **`POST /validate_batch`** |
| Processing model | Synchronous, single-threaded | Async + `ThreadPoolExecutor` (parallel) |
| FAISS index | Read from disk on every call | Cached in memory after first load |
| Fuzzy title list | Full DB scan on every call | Cached in memory after first load |
| S2 rate limiting | None | 1 req/s enforced via `threading.Lock` |
| Total timeout | None (Streamlit has 15s client timeout) | 14s hard cap; timed-out citations return `"TIMEOUT"` |

### How to switch between branches locally

```bash
# Switch to single-citation (master)
git checkout master

# Switch to multi-citation (this branch)
git checkout multi_citation
```

The `project_root/` contents are identical between branches. The differences are only in the code — `master` has the original sync pipeline, `multi_citation` adds batch + caching on top. Both run the same way:

```
uvicorn backend.api:app --port 8000          # backend (same command)
streamlit run frontend/app.py                 # frontend (same command)
```

---

## Architecture Differences

### `master` (single-citation)

```
User input (1 citation)
    |
    v
Streamlit  -->  POST /validate  -->  FastAPI (sync def)
    |                                      |
    |                              1. Parse citation
    |                              2. Fuzzy search (reads DB every call)
    |                              3. Semantic search (reads FAISS index every call)
    |                              4. Fuse + heuristic verify
    |                              5. [LLM verify]
    |                              6. [Live lookup - no rate limit]
    |                              7. Return single result
    v
Display single result card
```

### `multi_citation` (this branch)

```
User input (1+ citations, one per line)
    |
    v
Streamlit  -->  POST /validate_batch  -->  FastAPI (async def)
    |                                              |
    |                                      ThreadPoolExecutor
    |                                      +-- Thread 1: _verify_single(c1, batch_mode=True)
    |                                      |     |-- Fuzzy search (cached title list)
    |                                      |     |-- Semantic search (cached FAISS index)
    |                                      |     |-- [LLM verify - parallel]
    |                                      |     |-- [Live lookup - serialized at 1 req/s]
    |                                      |     `-- return result
    |                                      +-- Thread 2: _verify_single(c2, ...)
    |                                      +-- ...
    |                                      |
    |                                      asyncio.wait(futures, timeout=14s)
    |                                      cancel timed-out futures
    |                                      return {results: [...]}
    |
    v
Display: single result card OR summary table
```

### Key New Components

| Component | Location | Purpose |
|---|---|---|
| `_verify_single()` | `backend/api.py:83` | Extracted common verification logic used by both `/validate` and `/validate_batch` |
| `_acquire_s2_slot()` | `backend/api.py:39` | `threading.Lock`-based rate limiter ensuring ≤1 Semantic Scholar call/second |
| `_index_cache`, `_mapping_cache` | `backend/semantic_search.py` | FAISS index loaded once in memory, reused across all threads |
| `_title_cache` | `backend/fuzzy_search.py` | Full normalized-title list loaded once from DB, reused across all threads |
| `BatchValidateRequest` | `backend/api.py:31` | Pydantic model accepting `citations: list[str]` |
| `clear_index_cache()` | `backend/semantic_search.py` | Call after rebuilding FAISS index to force re-load |
| `clear_title_cache()` | `backend/fuzzy_search.py` | Call after re-ingesting DB to force re-load |

---

## New Config Variables

| Variable | Default | Description |
|---|---|---|
| `BATCH_TIMEOUT` | `14` | Max seconds for a batch request (Streamlit's client timeout is 15s) |
| `MAX_BATCH_SIZE` | `50` | Max citations per batch request |
| `S2_RATE_LIMIT` | `1.0` | Min seconds between Semantic Scholar API calls |

---

## Running the Pipeline

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic python-dotenv rapidfuzz faiss-cpu sentence-transformers numpy streamlit pandas
```

### 2. Ingest Paper Data

```bash
cd project_root/database
python ingest_openalex.py "machine learning"
```

### 3. Build FAISS Vector Index

```bash
cd project_root
python -c "from backend.semantic_search import build_faiss_index; build_faiss_index('papers.db', 'faiss_index.bin')"
```

### 4. Configure Environment

Edit `project_root/.env`:

```
DB_PATH=papers.db
FAISS_INDEX_PATH=faiss_index.bin
USE_LLM=true
DEEPSEEK_API_KEY=sk-...
USE_LIVE_LOOKUP=true
SEMANTIC_SCHOLAR_API_KEY=s2k-...
BATCH_TIMEOUT=14
MAX_BATCH_SIZE=50
S2_RATE_LIMIT=1.0
```

### 5. Start Backend

```bash
cd project_root
uvicorn backend.api:app --reload --port 8000
```

### 6. Start Frontend

```bash
cd project_root
streamlit run frontend/app.py
```

Open `http://localhost:8501`.

---

## API Endpoints

### POST /validate (unchanged — single citation)

```json
// Request
{"citation": "Smith, J. (2020). Machine learning. Journal of AI."}

// Response
{
  "label": "HALLUCINATED",
  "confidence": 0.9,
  "source": "llm_deepseek",
  "top_matches": [...],
  "reason": "...",
  "live_match": null
}
```

### POST /validate_batch (new — multi citation)

```json
// Request
{
  "citations": [
    "Smith, J. (2020). Machine learning. Journal of AI.",
    "Doe, A. (2019). Deep learning. NeurIPS."
  ]
}

// Response
{
  "results": [
    {
      "index": 0,
      "label": "HALLUCINATED",
      "confidence": 0.9,
      "source": "llm_deepseek",
      "top_matches": [...],
      "reason": "...",
      "live_match": null,
      "timed_out": false
    },
    {
      "index": 1,
      "label": "PARTIALLY_VALID",
      "confidence": 0.72,
      "source": "db_heuristic",
      "top_matches": [...],
      "reason": "...",
      "live_match": null,
      "timed_out": false
    }
  ]
}
```

Citations that exceed the 14s batch timeout return `"label": "TIMEOUT"` with `"timed_out": true`.

### GET /health

```json
{"status": "ok"}
```

---

## Cache Management

After re-ingesting the database or rebuilding the FAISS index, call the cache-clearing functions from the Python shell:

```python
from backend.semantic_search import clear_index_cache
from backend.fuzzy_search import clear_title_cache
clear_index_cache()
clear_title_cache()
```

Or simply restart the server — caches are re-loaded lazily on the first request.

---

## Running Tests

```bash
cd project_root
python -m pytest tests/ -v
```
