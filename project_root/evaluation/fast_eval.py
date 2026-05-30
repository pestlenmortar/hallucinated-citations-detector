"""
Final evaluation using real verifier code with pre-extracted features.
"""

import csv
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "training_dataset_full.csv")
PREDS_PATH = os.path.join(BASE_DIR, "results", "predictions_improved.csv")

LABELS = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; RS = "\033[0m"


def compute_metrics(preds):
    cm = defaultdict(lambda: defaultdict(int))
    for p in preds:
        cm[p["true_label"]][p["predicted_label"]] += 1
    mets = {}
    for l in LABELS:
        tp = cm[l].get(l, 0)
        fp = sum(cm[t][l] for t in LABELS if t != l)
        fn = sum(cm[l][t] for t in LABELS if t != l)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        mets[l] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    total_correct = sum(cm[l].get(l,0) for l in LABELS)
    mets["accuracy"] = total_correct/len(preds) if preds else 0
    mets["macro_f1"] = sum(mets[l]["f1"] for l in LABELS)/3
    mets["cm"] = cm
    return mets


def cf1(v):
    if v >= 0.85: return f"{G}{v:.4f}{RS}"
    if v >= 0.70: return f"{Y}{v:.4f}{RS}"
    return f"{R}{v:.4f}{RS}"


def main():
    from backend.classifier import invalidate_cache
    invalidate_cache()

    # Disable classifier for fast eval -- we don't have full top_candidate data
    from backend.verifier import verify_top_candidate, enable_classifier
    enable_classifier(False)
    print("Using real verifier (heuristic mode)")
    print("="*65)

    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    preds = []
    for i, row in enumerate(rows):
        cid = row["citation_id"]
        true_label = row["true_label"]
        etm = int(float(row.get("exact_title_match", 0)))
        edm = int(float(row.get("exact_doi_match", 0)))

        # Construct top_candidate from CSV features
        title_sim = float(row.get("title_similarity", 0))
        if title_sim > 0 or etm:
            top = {
                "fuzzy_score": title_sim * 100,
                "author_similarity": float(row.get("author_overlap", 0)),
                "doi_similarity": float(row.get("doi_match", 0)),
                "year_similarity": float(row.get("year_similarity", 0)),
                "venue_similarity": float(row.get("venue_similarity", 0)),
                "semantic_score": -1.0 if float(row.get("semantic_similarity", 0)) <= 0 else
                    (1.0 / float(row.get("semantic_similarity", 0))) - 1.0,
                "final_score": float(row.get("fusion_final_score", 0)),
                "metadata_score": float(row.get("metadata_score", 0)),
                "abstract_similarity": float(row.get("abstract_similarity", 0)),
                "paper_id": cid,
            }
        else:
            top = {}
        parsed = {"title": "", "authors": "", "year": None, "venue": "", "doi": ""}

        result = verify_top_candidate(top, parsed,
                                      exact_title_match=etm, exact_doi_match=edm)

        pred_label = result.get("label", "HALLUCINATED")
        conf = result.get("confidence", 0.0)
        correct = pred_label == true_label

        preds.append({
            "citation_id": cid, "true_label": true_label,
            "predicted_label": pred_label, "confidence": round(conf, 4),
            "corruption_type": row.get("corruption_type", ""),
            "notes": row.get("notes", ""), "correct": correct,
        })

        if (i+1) % 20 == 0:
            mets = compute_metrics(preds)
            print(f"  [{i+1:>3}/{len(rows)}]  "
                  f"V F1={cf1(mets['VALID']['f1'])}  "
                  f"PV F1={cf1(mets['PARTIALLY_VALID']['f1'])}  "
                  f"H F1={cf1(mets['HALLUCINATED']['f1'])}  "
                  f"Acc={mets['accuracy']:.3f}  mF1={cf1(mets['macro_f1'])}")

    mets = compute_metrics(preds)
    print(f"\n{'='*65}")
    print(f"{B}FINAL RESULTS{RS}")
    print(f"{'='*65}")

    all_pass = True
    for l in LABELS:
        m = mets[l]
        ok = m["f1"] >= 0.85
        if not ok: all_pass = False
        status = f"{G}PASS{RS}" if ok else f"{R}FAIL{RS}"
        print(f"  {l:<20} F1={cf1(m['f1'])}     P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"TP={m['tp']:<4} FP={m['fp']:<4} FN={m['fn']:<4} [{status}]")
    print(f"  {'-'*55}")
    print(f"  Acc={mets['accuracy']:.4f}  Macro F1={cf1(mets['macro_f1'])}")
    print(f"  {G}{B}ALL PASSED!{RS}" if all_pass else f"  {R}{B}Below 0.85{RS}")

    cm = mets["cm"]
    print(f"\n{B}Confusion Matrix:{RS}")
    print(f"  {'True \\ Pred':<20} {'VALID':>8} {'P_VALID':>8} {'HALL':>8}")
    for tl in LABELS:
        print(f"  {tl:<20} {cm[tl].get('VALID',0):>8} {cm[tl].get('PARTIALLY_VALID',0):>8} {cm[tl].get('HALLUCINATED',0):>8}")

    ct = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in preds:
        if p['true_label'] == 'PARTIALLY_VALID':
            ctype = p['corruption_type'] or 'unknown'
            ct[ctype]['total'] += 1
            if p['correct']: ct[ctype]['correct'] += 1
    print(f"\n{B}P_VALID by corruption type:{RS}")
    print(f"  {'Corruption':<25} {'Total':<6} {'Correct':<8} {'Acc':<8}")
    for ctype in sorted(ct):
        t = ct[ctype]['total']; c = ct[ctype]['correct']
        print(f"  {ctype:<25} {t:<6} {c:<8} {c/t:.2%}")

    os.makedirs(os.path.dirname(PREDS_PATH), exist_ok=True)
    with open(PREDS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["citation_id","true_label","predicted_label",
                                           "confidence","corruption_type","notes","correct"])
        w.writeheader()
        w.writerows(preds)
    print(f"\nPredictions: {PREDS_PATH} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
