"""
Build full 30-feature training dataset AND evaluate simultaneously.

Runs pipeline for each citation, saves features to training CSV,
computes metrics, shows live progress.

Usage:
    python evaluation/build_and_eval.py [--dry-run N]
"""

import argparse
import csv
import os
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.parser import parse_citation
from backend.normalization import normalize_title
from backend.feature_engineering import extract_feature_vector, FEATURE_NAMES
from backend.classifier import classify, is_model_available, invalidate_cache
from backend.verifier import verify_top_candidate
from backend.config import DB_PATH, FAISS_INDEX_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "test_citations.csv")
TRAINING_OUTPUT = os.path.join(BASE_DIR, "results", "training_dataset_full.csv")
PREDS_OUTPUT = os.path.join(BASE_DIR, "results", "predictions_improved.csv")

LABELS = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; D = "\033[2m"; RS = "\033[0m"

# Column 1: citation_id  2: true_label  3: pred_label  4: confidence
# 5: source  6: TP...etc  7: accuracy/macro

def load_dataset(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_one(raw, classifier_available):
    parsed = parse_citation(raw).model_dump()
    normed = normalize_title(parsed.get("title") or "")

    etm = 0; edm = 0
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT paper_id, title, authors, year, doi FROM papers WHERE normalized_title = ?",
            (normed,),
        ).fetchone()
        if row:
            etm = 1
            pid, db_title, db_authors, db_year, db_doi = row
            p_doi = (parsed.get("doi") or "").strip().lower()
            p_authors = (parsed.get("authors") or "").strip()
            p_year = parsed.get("year")

            if p_doi and db_doi and db_doi.strip().lower() == p_doi:
                edm = 1
                conn.close()
                return {"predicted_label": "VALID", "confidence": 1.0, "source": "db_exact",
                        "features": extract_feature_vector(
                            {"paper_id": pid, "title": db_title, "authors": db_authors,
                             "year": db_year, "doi": db_doi,
                             "fuzzy_score": 100.0, "semantic_score": 0.0,
                             "author_similarity": 1.0, "year_similarity": 1.0,
                             "venue_similarity": 0.0, "doi_similarity": 1.0,
                             "metadata_score": 0.5, "final_score": 100.0,
                             "abstract_similarity": 0.0},
                            parsed, etm, edm)}
            if not p_doi and p_authors and db_authors and p_year is not None and db_year is not None:
                from backend.fusion import _token_overlap, _year_similarity
                from backend.parser import ieee_author_overlap
                if _year_similarity(db_year, p_year) == 1.0:
                    overlap = max(
                        _token_overlap(p_authors, db_authors),
                        ieee_author_overlap(p_authors, db_authors),
                    )
                    if overlap >= 0.80:
                        conn.close()
                        return {"predicted_label": "VALID", "confidence": 1.0, "source": "db_exact",
                                "features": extract_feature_vector(
                                    {"paper_id": pid, "title": db_title, "authors": db_authors,
                                     "year": db_year, "doi": db_doi,
                                     "fuzzy_score": 100.0, "semantic_score": 0.0,
                                     "author_similarity": 1.0, "year_similarity": 1.0,
                                     "venue_similarity": 0.0, "doi_similarity": 0.0,
                                     "metadata_score": 0.25, "final_score": 100.0,
                                     "abstract_similarity": 0.0},
                                    parsed, etm, edm)}
        conn.close()
    except sqlite3.Error:
        pass

    exact_matches = []
    if etm:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT paper_id, title FROM papers WHERE normalized_title = ?", (normed,),
            ).fetchone()
            if row:
                exact_matches = [{"paper_id": row[0], "title": row[1], "score": 100.0}]
            conn.close()
        except sqlite3.Error:
            pass

    fuzzy = []
    try:
        from backend.fuzzy_search import fuzzy_search
        fuzzy = fuzzy_search(parsed.get("title") or normed, DB_PATH)
    except Exception:
        pass

    sem = []
    try:
        from backend.semantic_search import semantic_search
        sem = semantic_search(parsed.get("title") or normed, FAISS_INDEX_PATH)
    except Exception:
        pass

    fused = []
    try:
        from backend.fusion import fuse_candidates
        all_f = exact_matches + fuzzy
        fused = fuse_candidates(all_f, sem, parsed, DB_PATH)
    except Exception:
        pass

    top = fused[0] if fused else {}
    features = extract_feature_vector(top, parsed, etm, edm)

    if classifier_available and top:
        try:
            result = classify(top, parsed, exact_title_match=etm, exact_doi_match=edm)
            result["features"] = features
            result["source"] = result.get("source", "classifier")
            return result
        except Exception:
            pass

    result = verify_top_candidate(top, parsed, exact_title_match=etm, exact_doi_match=edm)
    result["features"] = features
    result["source"] = result.get("source", "heuristic")
    return result


def compute_metrics(preds):
    cm = defaultdict(lambda: defaultdict(int))
    for p in preds:
        cm[p["true_label"]][p["predicted_label"]] += 1
    mets = {}
    for label in LABELS:
        tp = cm[label].get(label, 0)
        fp = sum(cm[t][label] for t in LABELS if t != label)
        fn = sum(cm[label][t] for t in LABELS if t != label)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        mets[label] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    total_correct = sum(cm[l].get(l,0) for l in LABELS)
    mets["accuracy"] = total_correct/len(preds) if preds else 0
    mets["macro_f1"] = sum(mets[l]["f1"] for l in LABELS)/3
    return mets


def cf1(v):
    if v >= 0.85: return f"{G}{v:.4f}{RS}"
    if v >= 0.70: return f"{Y}{v:.4f}{RS}"
    return f"{R}{v:.4f}{RS}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=0, help="Limit to N records")
    args = parser.parse_args()

    rows = load_dataset(DATASET_PATH)
    if args.dry_run:
        rows = rows[:args.dry_run]
        print(f"{D}[DRY RUN] {len(rows)} records{RS}\n")

    total = len(rows)
    has_clf = is_model_available()
    print(f"Classifier: {G if has_clf else R}{has_clf}{RS}  |  Samples: {total}")
    print("="*65)

    preds = []
    training_rows = []
    start = time.time()

    for i, row in enumerate(rows, 1):
        cid = row["citation_id"]
        raw = row["raw_citation"]
        true_label = row["true_label"]

        t0 = time.time()
        result = run_one(raw, has_clf)
        elapsed = time.time() - t0

        pred_label = result.get("predicted_label", result.get("label", "ERROR"))
        conf = result.get("confidence", 0.0)
        correct = pred_label == true_label
        features = result.get("features", None)

        preds.append({
            "citation_id": cid, "raw_citation": raw,
            "true_label": true_label, "predicted_label": pred_label,
            "confidence": conf, "corruption_type": row.get("corruption_type",""),
            "notes": row.get("notes",""), "source": result.get("source",""),
            "correct": correct,
        })

        # Save features for training
        tr = {
            "citation_id": cid, "true_label": true_label,
            "corruption_type": row.get("corruption_type", ""),
            "notes": row.get("notes", ""),
        }
        if features is not None:
            for j, name in enumerate(FEATURE_NAMES):
                tr[name] = float(features[j]) if j < len(features) else 0.0
        else:
            for name in FEATURE_NAMES:
                tr[name] = 0.0
        training_rows.append(tr)

        # Display
        pct = i/total*100
        filled = int(30 * i/total)
        bar = "="*filled + "-"*(30-filled)
        status = f"{G}OK{RS}" if correct else f"{R}XX{RS}"
        eta = (time.time()-start)/i * (total-i) if i > 0 else 0
        print(f"\r[{i:>3}/{total}] [{bar}] {pct:5.1f}%  [{status}] {true_label:>5} -> {pred_label:<17}  {elapsed:.1f}s  ETA {eta:5.0f}s", end="", flush=True)

        if i % 20 == 0:
            mets = compute_metrics(preds)
            print(f"\n\n  {'Class':<20} {'F1':<10} {'P':<8} {'R':<8} {'TP':<5} {'FP':<5} {'FN':<5}")
            print(f"  {'-'*61}")
            for l in LABELS:
                m = mets[l]
                print(f"  {l:<20} {cf1(m['f1'])}     {m['precision']:.4f}   {m['recall']:.4f}   {m['tp']:<5} {m['fp']:<5} {m['fn']:<5}")
            print(f"  {'-'*61}")
            print(f"  {'Acc':<20} {mets['accuracy']:.4f}    {'Macro F1':<10} {cf1(mets['macro_f1'])}")

            # Save incremental training data
            fieldnames = ["citation_id", "true_label", "corruption_type", "notes"] + FEATURE_NAMES
            os.makedirs(os.path.dirname(TRAINING_OUTPUT), exist_ok=True)
            with open(TRAINING_OUTPUT, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(training_rows)

    # Final
    print(f"\n\n{'='*65}")
    print(f"{B}FINAL RESULTS{RESET}")
    print(f"{'='*65}")
    mets = compute_metrics(preds)

    all_pass = True
    for l in LABELS:
        m = mets[l]
        ok = m["f1"] >= 0.85
        if not ok: all_pass = False
        print(f"  {l:<20} F1={cf1(m['f1'])}     P={m['precision']:.4f}  R={m['recall']:.4f}  [{G}PASS{RS if ok else f'{R}FAIL{RS}'}]")
    print(f"  {'-'*55}")
    print(f"  Acc={mets['accuracy']:.4f}  Macro F1={cf1(mets['macro_f1'])}  Time={time.time()-start:.0f}s")
    print(f"  {G}{B}ALL PASSED!{RS}" if all_pass else f"  {R}{B}Below 0.85{RS}")

    # Save final outputs
    os.makedirs(os.path.dirname(TRAINING_OUTPUT), exist_ok=True)
    fieldnames = ["citation_id", "true_label", "corruption_type", "notes"] + FEATURE_NAMES
    with open(TRAINING_OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(training_rows)
    print(f"\nTraining data: {TRAINING_OUTPUT} ({len(training_rows)} rows)")

    pfieldnames = ["citation_id","raw_citation","true_label","predicted_label","confidence",
                   "corruption_type","notes","source","correct"]
    with open(PREDS_OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfieldnames)
        w.writeheader()
        w.writerows(preds)
    print(f"Predictions:  {PREDS_OUTPUT} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
