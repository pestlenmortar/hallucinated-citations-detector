"""
Direct evaluation — runs the full pipeline (retrieval + fusion + classifier/verifier
+ live lookup + LLM verification) locally without needing the API server.

Usage:
    python evaluation/direct_eval.py [--dry-run]
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.parser import parse_citation, ieee_author_overlap
from backend.normalization import normalize_title
from backend.feature_engineering import extract_feature_vector, FEATURE_NAMES
from backend.fusion import fuse_candidates, _token_overlap, _year_similarity
from backend.verifier import verify_top_candidate, heuristic_verify, llm_verify, llm_verify_direct
from backend.classifier import classify, is_model_available
from backend.config import DB_PATH, FAISS_INDEX_PATH, USE_LIVE_LOOKUP, USE_LLM
from backend.thresholding import DEFAULT_THRESHOLDS, apply_thresholds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "test_citations.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "predictions_improved.csv")

LABELS = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# LLM gate thresholds (lowered for broader coverage)
LLM_MATCH_Q_HIGH = 0.65
LLM_MATCH_Q_LOW = 0.45


def load_dataset(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(predictions: list[dict]) -> dict:
    cm = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        cm[p["true_label"]][p["predicted_label"]] += 1

    metrics = {}
    for label in LABELS:
        tp = cm[label].get(label, 0)
        fp = sum(cm[t][label] for t in LABELS if t != label)
        fn = sum(cm[label][t] for t in LABELS if t != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    total_correct = sum(cm[l].get(l, 0) for l in LABELS)
    total = len(predictions)
    metrics["accuracy"] = total_correct / total if total > 0 else 0.0
    metrics["macro_f1"] = sum(metrics[l]["f1"] for l in LABELS) / 3.0
    return metrics


def color_f1(f1: float) -> str:
    if f1 >= 0.85:
        return f"{GREEN}{f1:.4f}{RESET}"
    elif f1 >= 0.70:
        return f"{YELLOW}{f1:.4f}{RESET}"
    return f"{RED}{f1:.4f}{RESET}"


def run_single_citation(raw: str, true_label: str, classifier_available: bool) -> dict:
    """Run the FULL pipeline for a single citation (including LLM + live lookup)."""
    import sqlite3

    parsed = parse_citation(raw).model_dump()
    normed = normalize_title(parsed.get("title") or "")

    exact_title_match = 0
    exact_doi_match = 0

    # ── STEP 1: Exact match lookup ──
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

    # ── STEP 2: Strict exact match gate ──
    if exact_title_match:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT paper_id, title, authors, year, doi FROM papers WHERE normalized_title = ?",
                (normed,),
            ).fetchone()
            if row:
                _, db_title, db_authors, db_year, db_doi = row
                p_doi = (parsed.get("doi") or "").strip().lower()
                p_authors = (parsed.get("authors") or "").strip()
                p_year = parsed.get("year")

                if p_doi and db_doi and db_doi.strip().lower() == p_doi:
                    auth_overlap = max(
                        _token_overlap(p_authors, db_authors),
                        ieee_author_overlap(p_authors, db_authors),
                    ) if p_authors and db_authors else 0.0
                    year_ok = p_year is not None and db_year is not None and _year_similarity(db_year, p_year) == 1.0
                    if auth_overlap < 0.60 or not year_ok:
                        conn.close()
                        return {"predicted_label": "PARTIALLY_VALID", "confidence": 0.70, "source": "db_exact",
                                "reason": "DOI matches but metadata discrepancy (authors/year)"}
                    conn.close()
                    return {"predicted_label": "VALID", "confidence": 1.0, "source": "db_exact",
                            "reason": "Strict exact match (DOI)"}

                if not p_doi and p_authors and db_authors and p_year is not None and db_year is not None:
                    overlap = max(
                        _token_overlap(p_authors, db_authors),
                        ieee_author_overlap(p_authors, db_authors),
                    )
                    if _year_similarity(db_year, p_year) == 1.0 and overlap >= 0.80:
                        conn.close()
                        return {"predicted_label": "VALID", "confidence": 1.0, "source": "db_exact",
                                "reason": "Strict exact match (authors+year)"}
            conn.close()
        except sqlite3.Error:
            pass

    # ── STEP 3: Live lookup for semantic enrichment ──
    live_semantic_query = None
    live_lookup_cache = None
    if USE_LIVE_LOOKUP:
        try:
            from backend.live_lookup import live_lookup_verify
            live_result = live_lookup_verify(parsed)
            if live_result:
                live_lookup_cache = live_result
                abstract = (live_result.get("live_match") or {}).get("abstract") or ""
                if len(abstract) > len(parsed.get("title") or ""):
                    live_semantic_query = abstract
        except Exception:
            pass

    # ── STEP 4: Fuzzy + Semantic search ──
    fuzzy = []
    exact_matches = []
    try:
        from backend.fuzzy_search import fuzzy_search
        fuzzy = fuzzy_search(parsed.get("title") or normed, DB_PATH)
        if exact_title_match:
            try:
                conn = sqlite3.connect(DB_PATH)
                row = conn.execute(
                    "SELECT paper_id, title FROM papers WHERE normalized_title = ?",
                    (normed,),
                ).fetchone()
                if row:
                    exact_matches = [{"paper_id": row[0], "title": row[1], "score": 100.0}]
                conn.close()
            except sqlite3.Error:
                pass
    except (ImportError, Exception):
        pass

    sem = []
    try:
        from backend.semantic_search import semantic_search
        semantic_query = live_semantic_query or (parsed.get("title") or normed)
        sem = semantic_search(semantic_query, FAISS_INDEX_PATH)
    except (ImportError, FileNotFoundError, Exception):
        pass

    # ── STEP 5: Fusion ──
    fused = []
    try:
        all_fuzzy = exact_matches + fuzzy
        fused = fuse_candidates(all_fuzzy, sem, parsed, DB_PATH)
    except Exception:
        pass

    top = fused[0] if fused else {}

    # ── STEP 6: Classifier or heuristic verifier ──
    source = "classifier"
    if classifier_available and top:
        try:
            result = classify(top, parsed, exact_title_match=exact_title_match, exact_doi_match=exact_doi_match)
            source = result.get("source", "classifier")
        except Exception:
            result = verify_top_candidate(top, parsed, exact_title_match=exact_title_match,
                                          exact_doi_match=exact_doi_match)
            source = result.get("source", "heuristic")
    else:
        result = verify_top_candidate(top, parsed, exact_title_match=exact_title_match,
                                      exact_doi_match=exact_doi_match)
        source = result.get("source", "heuristic")

    # ── STEP 7: LLM verification (candidate-based) ──
    llm_used = False
    if USE_LLM and fused:
        top_c = fused[0]
        f_sim = top_c.get("fuzzy_score", 0.0) / 100.0
        s_raw = top_c.get("semantic_score", -1.0)
        s_sim = 1.0 / (1.0 + s_raw) if s_raw >= 0 else 0.0
        match_q = (f_sim + s_sim) / 2.0

        if match_q >= LLM_MATCH_Q_HIGH:
            try:
                llm_result = llm_verify(fused, parsed)
                if llm_result:
                    result = llm_result
                    source = "llm_deepseek"
                    llm_used = True
            except Exception:
                pass
        elif result.get("label") == "PARTIALLY_VALID" and match_q >= LLM_MATCH_Q_LOW:
            try:
                llm_result = llm_verify(fused, parsed)
                if llm_result and llm_result.get("label") == "VALID" and llm_result.get("confidence", 0.0) >= 0.80:
                    result = llm_result
                    source = "llm_deepseek"
                    llm_used = True
            except Exception:
                pass

    # ── STEP 8: Live lookup override ──
    if USE_LIVE_LOOKUP and not llm_used:
        result_label = result.get("label", "HALLUCINATED")
        result_conf = result.get("confidence", 0.0)

        if result_label == "HALLUCINATED" and result_conf >= 0.55:
            if live_lookup_cache:
                result = live_lookup_cache
                source = result.get("source", "live_lookup")
            else:
                try:
                    from backend.live_lookup import live_lookup_verify
                    live_result = live_lookup_verify(parsed)
                    if live_result:
                        result = live_result
                        source = result.get("source", "live_lookup")
                except Exception:
                    pass
        elif live_lookup_cache and live_lookup_cache.get("label") == "VALID":
            ll_score = live_lookup_cache.get("confidence", 0.0)
            if ll_score >= 0.80:
                live_lookup_cache["source"] = "live_lookup"
                return {
                    "predicted_label": live_lookup_cache.get("label", "HALLUCINATED"),
                    "confidence": live_lookup_cache.get("confidence", 0.0),
                    "source": "live_lookup",
                    "reason": live_lookup_cache.get("reason", ""),
                }

    # ── STEP 9: Direct LLM fallback ──
    if USE_LLM and result.get("label") == "HALLUCINATED":
        try:
            llm_result = llm_verify_direct(parsed)
            if llm_result:
                result = llm_result
                source = "llm_deepseek"
        except Exception:
            pass

    return {
        "predicted_label": result.get("label", "HALLUCINATED"),
        "confidence": result.get("confidence", 0.0),
        "source": source,
        "reason": result.get("reason", ""),
    }


def main(dry_run: bool = False):
    rows = load_dataset(DATASET_PATH)
    if dry_run:
        rows = rows[:30]
        print(f"{DIM}[DRY RUN] First 30 records{RESET}\n")

    total = len(rows)
    has_classifier = is_model_available()
    print(f"Classifier available: {GREEN if has_classifier else RED}{has_classifier}{RESET}")
    print(f"LLM enabled: {GREEN if USE_LLM else RED}{USE_LLM}{RESET}")
    print(f"Live lookup enabled: {GREEN if USE_LIVE_LOOKUP else RED}{USE_LIVE_LOOKUP}{RESET}")
    print(f"Evaluating {total} citations...\n")

    predictions = []
    start_time = time.time()

    for i, row in enumerate(rows, 1):
        cid = row["citation_id"]
        raw = row["raw_citation"]
        true_label = row["true_label"]
        corruption_type = row.get("corruption_type", "")
        notes = row.get("notes", "")

        t0 = time.time()
        result = run_single_citation(raw, true_label, has_classifier)
        elapsed = time.time() - t0

        pred_label = result["predicted_label"]
        confidence = result["confidence"]
        correct = pred_label == true_label

        predictions.append({
            "citation_id": cid,
            "raw_citation": raw,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "corruption_type": corruption_type,
            "notes": notes,
            "source": result["source"],
            "correct": correct,
        })

        pct = i / total * 100
        bar_width = 30
        filled = int(bar_width * i / total)
        bar = "=" * filled + "-" * (bar_width - filled)
        status = f"{GREEN}OK{RESET}" if correct else f"{RED}XX{RESET}"
        eta = (time.time() - start_time) / i * (total - i) if i > 0 else 0

        print(f"\r[{i:>3}/{total}] [{bar}] {pct:5.1f}%  "
              f"[{status}] {true_label:>5} -> {pred_label:<17}  "
              f"src={result['source'][:8]:<8} {elapsed:.1f}s  ETA {eta:>5.0f}s", end="", flush=True)

        if i > 0 and i % 10 == 0:
            metrics = compute_metrics(predictions)
            print(f"\n\n  {'Class':<20} {'F1':<10} {'P':<10} {'R':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
            print(f"  {'-'*68}")
            for label in LABELS:
                m = metrics[label]
                f1_disp = color_f1(m["f1"])
                print(f"  {label:<20} {f1_disp}     {m['precision']:.4f}     {m['recall']:.4f}     {m['tp']:<6} {m['fp']:<6} {m['fn']:<6}")
            print(f"  {'-'*68}")
            print(f"  {'Accuracy':<20} {metrics['accuracy']:.4f}")
            print(f"  {'Macro F1':<20} {color_f1(metrics['macro_f1'])}")
            print()

    # ── Final results ──
    print(f"\n\n{'='*70}")
    print(f"{BOLD}FINAL RESULTS{RESET}")
    print(f"{'='*70}")
    metrics = compute_metrics(predictions)

    all_pass = True
    for label in LABELS:
        m = metrics[label]
        f1_disp = color_f1(m["f1"])
        status = f"{GREEN}PASS{RESET}" if m["f1"] >= 0.85 else f"{RED}FAIL{RESET}"
        if m["f1"] < 0.85:
            all_pass = False
        print(f"  {label:<20} F1={f1_disp}     P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"TP={m['tp']:<4} FP={m['fp']:<4} FN={m['fn']:<4} [{status}]")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<20} {metrics['accuracy']:.4f}")
    print(f"  {'Macro F1':<20} {color_f1(metrics['macro_f1'])}")
    print(f"  {'Total samples':<20} {len(predictions)}")
    print(f"  {'Total time':<20} {time.time() - start_time:.1f}s")

    if all_pass:
        print(f"\n  {GREEN}{BOLD}ALL CLASSES PASSED F1 >= 0.85!{RESET}")
    else:
        print(f"\n  {RED}{BOLD}SOME CLASSES BELOW 0.85 -- see analysis above{RESET}")

    # Confusion matrix
    cm = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        cm[p["true_label"]][p["predicted_label"]] += 1
    print(f"\n{BOLD}Confusion Matrix:{RESET}")
    print(f"  True \\ Pred    HALL   P_VALID  VALID")
    for tl in LABELS:
        row_str = f"  {tl:<15}"
        for pl in LABELS:
            row_str += f" {cm[tl].get(pl, 0):>7}"
        print(row_str)

    # Source breakdown
    src_counts = defaultdict(int)
    for p in predictions:
        src_counts[p.get("source", "unknown")] += 1
    print(f"\n{BOLD}Source Breakdown:{RESET}")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    # Write results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = [
        "citation_id", "raw_citation", "true_label", "predicted_label",
        "confidence", "corruption_type", "notes", "source", "correct",
    ]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"\nPredictions written to {RESULTS_PATH}")

    return predictions, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run on first 10 records")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
