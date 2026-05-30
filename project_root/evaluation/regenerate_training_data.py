"""
Regenerate the training dataset with all 30+ features.

Runs the retrieval+fusion pipeline directly for each citation
in evaluation/datasets/test_citations.csv, extracts the full feature
vector, and writes evaluation/results/training_dataset_full.csv.

Usage:
    python evaluation/regenerate_training_data.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.parser import parse_citation
from backend.normalization import normalize_title
from backend.feature_engineering import extract_feature_vector, FEATURE_NAMES
from backend.config import DB_PATH, FAISS_INDEX_PATH, USE_LIVE_LOOKUP
from backend import config as backend_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "test_citations.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "training_dataset_full.csv")


def load_dataset(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_pipeline_for_citation(raw: str) -> dict | None:
    """Run retrieval+fusion and return top candidate + match flags."""
    parsed = parse_citation(raw).model_dump()
    normed = normalize_title(parsed.get("title") or "")

    exact_title_match = 0
    exact_doi_match = 0

    # Check exact title match
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT paper_id, title, doi FROM papers WHERE normalized_title = ?",
            (normed,),
        ).fetchone()
        if row:
            exact_title_match = 1
            db_doi = (row[2] or "").strip().lower()
            p_doi = (parsed.get("doi") or "").strip().lower()
            if p_doi and db_doi and p_doi == db_doi:
                exact_doi_match = 1
        conn.close()
    except sqlite3.Error:
        pass

    # Fuzzy search
    fuzzy = []
    try:
        from backend.fuzzy_search import fuzzy_search
        fuzzy = fuzzy_search(parsed.get("title") or normed, DB_PATH)
    except (ImportError, Exception):
        pass

    # Live lookup for abstract
    live_abstract = None
    if USE_LIVE_LOOKUP:
        try:
            from backend.live_lookup import live_lookup_verify
            live_result = live_lookup_verify(parsed)
            if live_result:
                live_abstract = (live_result.get("live_match") or {}).get("abstract") or ""
        except Exception:
            pass

    # Semantic search
    sem = []
    try:
        from backend.semantic_search import semantic_search
        query = live_abstract or (parsed.get("title") or normed)
        sem = semantic_search(query, FAISS_INDEX_PATH)
    except (ImportError, FileNotFoundError, Exception):
        pass

    # Fusion
    fused = []
    try:
        from backend.fusion import fuse_candidates
        all_fuzzy = [{"paper_id": r["paper_id"], "title": r.get("title", ""), "score": r.get("score", 0)}
                     for r in fuzzy]
        fused = fuse_candidates(all_fuzzy, sem, parsed, DB_PATH)
    except Exception:
        pass

    top = fused[0] if fused else {}
    return {
        "parsed": parsed,
        "top_candidate": top,
        "exact_title_match": exact_title_match,
        "exact_doi_match": exact_doi_match,
    }


def main():
    rows = load_dataset(DATASET_PATH)
    total = len(rows)

    # Check if partial file exists
    fieldnames = ["citation_id", "true_label", "corruption_type", "notes"] + FEATURE_NAMES
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    start_i = 0
    mode = "w"

    with open(OUTPUT_PATH, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for i, row in enumerate(rows, 1):
            cid = row["citation_id"]
            true_label = row["true_label"]
            corruption_type = row.get("corruption_type", "")
            notes = row.get("notes", "")

            result = run_pipeline_for_citation(row["raw_citation"])
            features = []
            if result and result.get("top_candidate"):
                feature_vec = extract_feature_vector(
                    result["top_candidate"],
                    result["parsed"],
                    exact_title_match=result["exact_title_match"],
                    exact_doi_match=result["exact_doi_match"],
                )
                features = [float(v) for v in feature_vec]
            else:
                features = [0.0] * len(FEATURE_NAMES)

            out_row = {
                "citation_id": cid,
                "true_label": true_label,
                "corruption_type": corruption_type,
                "notes": notes,
            }
            for j, name in enumerate(FEATURE_NAMES):
                out_row[name] = features[j] if j < len(features) else 0.0

            writer.writerow(out_row)

            if i % 10 == 0:
                print(f"[{i}/{total}] Processed {i} citations")

    print(f"\nWrote {total} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
