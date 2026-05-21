# Citation Validator

A full-stack academic citation verification system that parses raw citation strings (APA/IEEE), searches a local database using fuzzy and semantic matching, fuses candidates with weighted scoring, and returns a verdict (VALID / PARTIALLY_VALID / HALLUCINATED) via heuristic or LLM-based verification.

## Architecture

```
User Browser (Streamlit frontend)
    |
    | POST /validate {"citation": "..."}
    v
FastAPI Backend
    |
    |-- parser.py        -> models.py (ParsedCitation)
    |-- normalization.py -> normalize_title()
    |-- fuzzy_search.py  -> RapidFuzz + SQLite
    |-- semantic_search.py -> Sentence-Transformers + FAISS
    |-- fusion.py        -> weighted scoring of candidates
    |-- verifier.py      -> heuristic and/or LLM verification
    |                       (Ollama + qwen2.5:3b)
    |
    v
SQLite (papers.db) <-- ingest_openalex.py (OpenAlex API)
```

## Project Structure

```
project_root/
├── backend/
│   ├── api.py              # FastAPI entry point
│   ├── config.py           # .env configuration loader
│   ├── fusion.py           # Candidate fusion and scoring
│   ├── fuzzy_search.py     # RapidFuzz-based fuzzy matching
│   ├── models.py           # Pydantic data models
│   ├── normalization.py    # Text normalization utilities
│   ├── parser.py           # Citation string parser (APA/IEEE)
│   ├── retrieval.py        # (stub) retrieval orchestration
│   ├── scoring.py          # (stub) scoring algorithms
│   ├── semantic_search.py  # FAISS + Sentence-Transformers search
│   ├── utils.py            # (stub) shared utilities
│   └── verifier.py         # Heuristic and LLM verifier
├── database/
│   ├── schema.sql          # SQLite table DDL
│   ├── ingest_openalex.py  # OpenAlex API data ingestion
│   └── ingest_semanticscholar.py  # (stub) Semantic Scholar ingestion
├── embeddings/
│   └── build_index.py      # (stub) vector index builder
├── frontend/
│   ├── app.py              # Streamlit main page
│   ├── pages/
│   │   ├── analytics.py    # (stub) analytics page
│   │   └── results.py      # Detailed results page
│   └── styles/
│       └── minimal.css     # Custom Streamlit styles
├── llm/
│   ├── inference.py        # (stub) LLM inference wrapper
│   ├── json_schema.py      # Pydantic schema for LLM output
│   └── prompts/
│       └── verification_prompt.txt  # LLM verification prompt
├── tests/
│   ├── test_fusion.py      # Fusion module tests
│   ├── test_parser.py      # Citation parser tests
│   ├── test_retrieval.py   # (stub) retrieval tests
│   └── test_verifier.py    # Verifier module tests
├── .env                    # Environment configuration
├── docker-compose.yml      # (stub) Docker orchestration
├── papers.db               # SQLite database
└── requirements.txt        # Python dependencies
```

## Requirements

### Python Dependencies

- fastapi
- uvicorn
- pydantic
- python-dotenv
- rapidfuzz
- faiss-cpu
- sentence-transformers
- numpy
- streamlit
- pandas

### External Services

- **Ollama** (optional) -- required only if LLM-based verification is enabled. The default model is `qwen2.5:3b`.

## Setup and Usage

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic python-dotenv rapidfuzz faiss-cpu sentence-transformers numpy streamlit pandas
```

### 2. (Optional) Pull LLM Model

```bash
ollama pull qwen2.5:3b
```

### 3. Ingest Paper Data

Fetch papers from OpenAlex by keyword:

```bash
cd project_root/database
python ingest_openalex.py "machine learning"
```

This populates `papers.db` with paper metadata from the OpenAlex API.

### 4. Build FAISS Vector Index

```bash
cd project_root
python -c "from backend.semantic_search import build_faiss_index; build_faiss_index('papers.db', 'faiss_index.bin')"
```

### 5. Configure Environment

Create or edit `project_root/.env`:

```
DB_PATH=papers.db
FAISS_INDEX_PATH=faiss_index.bin
USE_LLM=false
OLLAMA_MODEL=qwen2.5:3b
```

Set `USE_LLM=true` to enable verification via Ollama.

### 6. Start the Backend

```bash
cd project_root
uvicorn backend.api:app --reload --port 8000
```

### 7. Start the Frontend

In a separate terminal:

```bash
cd project_root
streamlit run frontend/app.py
```

### 8. Use the Application

Open `http://localhost:8501` in your browser, paste a citation string (APA or IEEE format), and click **Validate**. The system will:

1. Parse the citation to extract metadata
2. Search the database using exact, fuzzy, and semantic matching
3. Fuse candidates into a ranked list with weighted scores
4. Verify the top candidate and return a verdict
5. Display the result with a color-coded card and top matches

**Verdict meanings:**
- **VALID** -- citation matches a known paper with high confidence
- **PARTIALLY_VALID** -- citation partially matches but has discrepancies
- **HALLUCINATED** -- no matching paper found in the database

## Running Tests

```bash
cd project_root
python -m pytest tests/ -v
```

## API Endpoints

### POST /validate

Request body:
```json
{
  "citation": "Smith, J., & Doe, A. (2020). Machine learning is great. Journal of AI, 15(3), 123-145."
}
```

Response:
```json
{
  "label": "VALID",
  "confidence": 0.95,
  "reason": "High match confidence",
  "top_matches": [...]
}
```

### GET /health

Returns `{"status": "ok"}`.
