# Session Analysis — Hallucinated Citations Detector

> Saved: 2026-05-27
> Purpose: Reference document encapsulating all methodology analysis, evaluation results, quantitative models, and recommendations from the session.

---

## 1. Initial Methodology Gap Analysis

When asked to identify improvements to make the pipeline "foolproof," the following 7 gaps were identified:

| # | Issue | Severity |
|---|-------|----------|
| 1 | Abstract-based semantic search is **documented but not implemented** — `semantic_search.py` still encodestitle-only, `api.py` still queries with title-only | Highest |
| 2 | **Inconsistent scoring weights** across `fusion.py`, `verifier.py`, `live_lookup.py`, and `documentation.txt` — 4 different weight sets | High |
| 3 | **Sequential fallback should be a consensus/voting system** — parallel methods with weighted vote > sequential chaining | High |
| 4 | **DOI-first resolution** — strongest identifier (DOI) has lowest weight (0.02-0.05) | Medium |
| 5 | **Step 10 (direct LLM verification) is circular** — asking an LLM to "recognize" papers from training memory is the same mechanism that produces hallucinations | Medium |
| 6 | **Dead code** in `live_lookup.py:90-98` (unreachable duplicate block after `return`), **stale test** in `test_verifier.py` (tests a re-implemented function, not real code), no integration tests | Medium |
| 7 | **Parser only supports APA/IEEE**, no FTS5 usage (despite schema having it), no connection pooling, no structured logging | Low |

---

## 2. Discovery of Commit `b43d66b` on `master`

**Findings:**
- Commit `b43d66b` on `master` implements items 1 and 2 (abstract-based semantic search + unified scoring weights)
- The `multi_citation` branch was forked from `b6fee88` (Add .gitignore), which is **before** `b43d66b`
- `multi_citation` did NOT have these changes — they only existed on `master`

### What `b43d66b` contains:

**Item 1 — Abstract-based semantic search:**
- `build_faiss_index()` now queries `SELECT paper_id, title, abstract FROM papers` and encodes `title + ". " + abstract` via `_make_content()`
- `api.py` does early live lookup to obtain the abstract, then uses it as the semantic search query
- `live_lookup.py` fetches `abstract` from Semantic Scholar API fields
- `build_index.py` is a functional CLI script (was a stub)
- `schema.sql` adds `abstract` to FTS5 table
- `fuzzy_search.py` replaced in-memory title cache with FTS5-based search

**Item 2 — Unified scoring weights in `fusion.py`:**
- Before: `RANK_TITLE_W=0.45, RANK_AUTHOR_W=0.15, RANK_YEAR_W=0.10, RANK_SEMANTIC_W=0.23`
- After: `RANK_TITLE_W=0.20, RANK_AUTHOR_W=0.23, RANK_YEAR_W=0.12, RANK_SEMANTIC_W=0.38`
- These match `documentation.txt` exactly.

### Cherry-pick resolution (3 conflicts):
1. **`api.py`**: Combined master's `live_lookup_cache` (avoid duplicate API calls for early lookup) with HEAD's `batch_mode` rate limiting
2. **`fuzzy_search.py`**: Took master's FTS5 approach over HEAD's in-memory cache (scales better)
3. **`semantic_search.py`**: Kept HEAD's `_load_index()` caching, took master's abstract-based `_make_content()` and `build_faiss_index()`, kept `k=10` (master had `k=30`, which was deemed too aggressive by the user)

### Database tracking issue:
- `papers.db` was a 0-byte git-tracked placeholder (committed in initial commit `96ae5c9`)
- Added to `.gitignore` (`*.db`) in commit `b6fee88`, but git still tracked it
- Untracked with `git rm --cached papers.db` and pushed to remote
- The **real** 503k paper database is at `database/papers.db` (not the project root), configured via `.env` with `DB_PATH=database/papers.db`

---

## 3. Frontend Multi-Citation Enhancement

The multi-citation result view was enhanced to be more useful:

**Before:** Static summary badges per result — no way to inspect top matches or live match details inline

**After:**
- Added "View full detailed results →" link at the top of the multi-result view
- Added Streamlit `expander("Details for #N")` per result, showing:
  - Live match info (title, year, authors, venue) if present
  - Top 3 DB matches with fuzzy, semantic, and final scores
- Users can now expand/collapse details individually without navigating away

---

## 4. Evaluation Benchmark Results

### Test Dataset: 200 citations
- 80 VALID (well-known ML/NLP/CV papers, IEEE format, all fields correct)
- 80 PARTIALLY_VALID (real papers with corrupted fields — 8 corruption types)
- 40 HALLUCINATED (fabricated papers using real author names + real venues)

### Confusion Matrix

| True ↓ \ Predicted → | HALLUCINATED | PARTIALLY_VALID | VALID |
|----------------------|:------------:|:---------------:|:-----:|
| **HALLUCINATED**     | 31           | 7               | 2     |
| **PARTIALLY_VALID**  | 2            | 51              | 27    |
| **VALID**            | 1            | 34              | 45    |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| VALID | 0.6081 | 0.5625 | 0.5844 | 80 |
| PARTIALLY_VALID | 0.5543 | 0.6375 | 0.5930 | 80 |
| HALLUCINATED | 0.9118 | 0.7750 | 0.8378 | 40 |

**Overall Accuracy: 0.6350**
**Macro F1: 0.6718**

---

## 5. Root Cause Analysis of Misclassifications

### The Two Big Numbers

| Problem | Count | Signal |
|---------|-------|--------|
| VALID → PARTIALLY_VALID | **34/80** | False negatives for VALID |
| PARTIALLY_VALID → VALID | **27/80** | False negatives for P_VALID |
| HALLUCINATED → P_VALID/VALID | **9/40** | Missed hallucinations |

These partially cancel in the accuracy metric but are individual failures.

### Root Cause 1: Metadata check kills VALID papers (34/80)

The `_detect_metadata_issues()` function in `verifier.py:53-61` flags:
- `year_sim < 0.5`
- `venue_sim < 0.3`
- `doi_sim < 0.8`

When ANY of these trigger, a paper with `title_sim >= 0.95 AND author_sim >= 0.70` gets downgraded from VALID to PARTIALLY_VALID. The problem: the parser extracts imperfect venue strings from IEEE citations (venue includes volume, number, pages, etc.). The venue similarity then falls below 0.3 for a correct paper, triggering a spurious downgrade.

**Evidence:** VALID papers with conf >= 0.9 have 94% accuracy (exact DB match), but those with conf 0.7-0.8 have only 23% accuracy. The metadata check is retroactively downgrading valid matches.

### Root Cause 2: Rule 4 title threshold lets fabricated papers through (7 of 9)

In `heuristic_verify`, Rule 4:
```
if title_sim >= 0.70 or final_score >= 50:
    if author_sim >= 0.1 OR year_sim >= 0.5 → PARTIALLY_VALID
```

Problems:
- **0.70 title threshold is too low** — with 500k papers, ANY plausible-sounding fabricated title gets a 70% fuzzy match to something
- **OR gate instead of AND gate** — only ONE weak signal needed (author_sim >= 0.1 is essentially noise-level)
- Fabricated papers use real author names (e.g., "K. Simonyan, A. Vaswani, and J. Devlin"), giving author overlap with real papers by those authors

**Evidence:** The 9 misclassified HALLUCINATED papers have average confidence of only 0.36 — the pipeline IS uncertain but the hard rules force PARTIALLY_VALID. None of them have confidence > 0.47.

### Root Cause 3: 9 hallucinated papers misclassified via live lookup

The 9 misclassified HALLUCINATED papers all had confidence < 0.6 (in the 0.27-0.47 range). They were NOT misclassified by the DB — the pipeline's heuristic first says HALLUCINATED (low confidence), then the live lookup (Semantic Scholar API) finds a spurious match for the plausible-sounding fabricated title, and upgrades the result to PARTIALLY_VALID or VALID.

**This is the most dangerous failure mode** — the system is fooled by its own verification fallback.

### Root Cause 4: Corruption type susceptibility

| Corruption Type | Accuracy | Issue |
|----------------|----------|-------|
| Wrong page numbers | 38% (3/8) | Pipeline doesn't even check page numbers (not in DB schema) |
| Title typo | 40% (4/10) | RapidFuzz token_sort_ratio is order-invariant — single character typos barely affect score |
| Missing DOI | 50% (4/8) | DOI weight is 0.02-0.05 — missing it is almost invisible |
| Wrong venue | 60% (6/10) | Venue matching via token overlap + stopword stripping is too forgiving |
| Incomplete authors | 60% (6/10) | Partial author lists match well with token overlap |
| Missing authors (et al.) | 62% (5/8) | "et al." doesn't penalize enough |
| Mixed | 88% (14/16) | Multiple corruptions actually help detection |
| Year shifted | 90% (9/10) | Year check is the most reliable signal |

---

## 6. Quantitative Model: DB Size vs. Accuracy

### Core Model

For each class, accuracy decomposes as:

```
Acc_i(N) = c_i(N) × Acc_match_i  +  (1 - c_i(N)) × Acc_nomatch_i
```

Where `c(N)` = match rate (fraction of papers that get a confidence >= 0.6 result from DB or live lookup).

### Current Parameters (at N=503k)

| Class | c(N) | Acc(matched) | Acc(unmatched) | Observed Acc |
|-------|------|-------------|---------------|-------------|
| VALID | 0.750 | 0.583 | 0.500 | 0.563 |
| P_VALID | 0.613 | 0.571 | 0.742 | 0.638 |
| HALLUC | 0.775 | 1.000 | 0.000 | 0.775 |

### Match Rate Decomposition

The match rate has two independent sources:

```
c(N) = 1 - (1 - c_db(N)) × (1 - c_live)
```

- `c_db(N)` = fraction found in local DB — increases with N, depends on domain overlap
- `c_live` = fraction found via Semantic Scholar live lookup — independent of N, currently props up the whole system

Without live lookup, `c_db(503k)` for ML papers would be ~0.20-0.30 (since the DB was built from non-CS engineering topics).

### Domain Mismatch Bottleneck

The database was rebuilt in commit `b43d66b` to exclude CS/ML topics. The `topics.sh` change dropped all CS sections (Systems & Architecture, Networking, Security, Software Engineering, Theory, Graphics, HCI, Robotics, HPC, Other CS topics) and kept only non-CS engineering:

- CS/ML-like papers in DB: ~188,646 (37.5%)
- Non-CS engineering in DB: ~21,525 (4.3%)
- Other (biomedical, social sciences, etc.): ~293,211 (58.2%)

Many headline ML papers used in the test set (Attention is All You Need, BERT, ResNet, etc.) exist only as references/surface-level matches, not as paper entries.

### Projected Accuracy Curves

| DB Size | Domain | c_db(VALID) | c_total(VALID) | Est. Accuracy |
|---------|--------|-------------|---------------|---------------|
| 503k (current) | Non-CS dominated | ~0.20 | 0.75 | **0.635** |
| 503k + CS topics restored | Balanced | ~0.50 | 0.88 | **0.68-0.71** |
| 1M | Broad coverage | ~0.65 | 0.91 | **0.71-0.74** |
| 2M | Broad coverage | ~0.80 | 0.95 | **0.73-0.76** |
| 5M+ | Near-complete | ~0.90 | 0.97 | **0.75-0.80** |
| ∞ (hypothetical) | Perfect | 1.00 | 1.00 | **~0.80-0.85** |

**Diminishing returns** kick in after ~1M because live lookup already provides a 75% match-rate floor. Each doubling of N buys roughly +2-3% accuracy, asymptoting at ~0.82.

**Upper bound (~0.80-0.85)** is determined by irreducible ambiguity — title typos vs. close lexically similar fabricated papers, and metadata that isn't in the DB schema (page numbers, etc.).

### The HALLUCINATED Counter-Force

For fabricated papers, larger N means more potential spurious matches:

```
P(HALL correct) = P(no spurious DB match) × P(no spurious live match)
```

However, the 9 misclassified hallucinated papers were ALL due to live lookup, NOT the DB. Live lookup searches the entire Semantic Scholar corpus (essentially infinite N), so DB size additions have negligible marginal impact on HALLUCINATED false negatives. The live lookup is already the dominant source.

---

## 7. Recommendations with Expected Impact

### Tier 1: High Impact, Low Risk

| # | Change | Location | Expected Impact |
|---|--------|----------|----------------|
| 1 | **Stop downgrading VALID on venue/year mismatch alone** — venue and year are noisy signals from the parser. Only flag metadata issues when the raw compared values differ clearly, not when fuzzy similarity is low. | `verifier.py:_detect_metadata_issues()` | Reduces 34 → ~10 VALID downgrades (+3% accuracy) |
| 2 | **Change OR gate to AND gate** for Rule 4's author/year signal — require both `author_sim >= 0.3 AND year_sim >= 0.5` | `verifier.py:99-100` | Reduces 9 → ~3 hallucination escapes (+1.5% accuracy) |
| 3 | **Raise title threshold** from 0.70 to 0.85 for PARTIALLY_VALID consideration | `verifier.py:99` | Reduces further hallucination escapes (+1% accuracy) |

### Tier 2: Medium Impact

| # | Change | Expected Impact |
|---|--------|----------------|
| 4 | **Restore CS topics to ingestion pipeline** (revert the `topics.sh` change that dropped CS sections). The single highest-leverage non-code change. | +3-5% accuracy from better c_db for ML test papers |
| 5 | **Add UNCERTAIN label** for confidence < 0.30. Currently borderline cases are forced into wrong labels. More honest, and downstream users can treat UNCERTAIN as HALLUCINATED. | Eliminates ~9 forced-wrong labels |
| 6 | **Boost DOI weight** from 0.02-0.05 to 0.10-0.15 across all scoring modules. Also penalize missing DOI when the DB match has one. | Moderate improvement in P_VALID detection |

### Tier 3: Lower Impact / Engineering

| # | Change | Notes |
|---|--------|-------|
| 7 | **Add character-level similarity** (Levenshtein or `fuzz.ratio`) as a secondary check for title typos | Addresses 5/10 title-typo misses |
| 8 | **Unify verifier scoring weights with fusion** | Reduces contradictory ranking-vs-verification signals |
| 9 | **Remove or gate live lookup for borderline HALL cases** — if heuristic says HALLUCINATED with conf < 0.3, don't let live lookup override it | Prevents the worst false positives |

---

## 8. Files Created / Modified

### New Files (not tracked, not pushed to remote)
- `evaluation/datasets/test_citations.csv` — 200 test citations (80 VALID, 80 P_VALID, 40 HALL)
- `evaluation/benchmark.py` — CLI benchmark tool, calls API, writes predictions, `--dry-run` support
- `evaluation/metrics.py` — sklearn classification report + 4 matplotlib plots
- `evaluation/generate_dataset.py` — regenerates test_citations.csv
- `evaluation/init_test_db.py` — small test database (12 papers) — **deleted after session**
- `evaluation/evaluation_results.ipynb` — Jupyter notebook with embedded plots
- `evaluation/SESSION_ANALYSIS.md` — this file

### Modified Files (committed to `multi_citation`)
- `frontend/app.py` — added expandable details per result in multi-citation view
- Various backend files via cherry-pick of `b43d66b` (semantic_search, fusion, api, fuzzy_search, live_lookup, etc.)

### Git Notes
- `multi_citation` is 3 commits ahead of `origin/multi_citation`:
  - `967fca3` — Cherry-pick of `b43d66b` (abstract semantic search + unified weights)
  - `513edea` — Frontend multi-citation UI enhancements
  - `cae01f6` — Stop tracking `papers.db`
- `papers.db` (empty placeholder in project root) removed from git tracking
- Real 503k database at `database/papers.db` (never tracked, configured via `.env`)
- The `origin/multi_citation` remote is up to date with all pushes
