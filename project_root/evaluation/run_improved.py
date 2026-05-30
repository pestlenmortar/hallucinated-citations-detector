"""
Improved evaluation benchmark with live processing display.

Runs the citation validation API against test_citations.csv,
shows per-class running stats, and writes results to
evaluation/results/predictions_improved.csv

Usage:
    python evaluation/run_improved.py [--wait SECONDS]
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "test_citations.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "predictions_improved.csv")
API_URL = "http://localhost:8000/validate"
TIMEOUT = 30
LABELS = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 100


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def _compute_f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def load_dataset(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def call_api(citation: str) -> dict | None:
    payload = json.dumps({"citation": citation}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def compute_running_metrics(predictions: list[dict]) -> dict:
    """Compute per-class precision, recall, F1 from accumulated predictions."""
    cm = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        true = p["true_label"]
        pred = p["predicted_label"]
        cm[true][pred] += 1

    metrics = {}
    for label in LABELS:
        tp = cm[label].get(label, 0)
        fp = sum(cm[t][label] for t in LABELS if t != label)
        fn = sum(cm[label][t] for t in LABELS if t != label)
        f1 = _compute_f1(tp, fp, fn)
        metrics[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    total_correct = sum(cm[l].get(l, 0) for l in LABELS)
    total = len(predictions)
    metrics["accuracy"] = round(total_correct / total, 4) if total > 0 else 0.0

    macro_f1 = sum(metrics[l]["f1"] for l in LABELS) / 3.0
    metrics["macro_f1"] = round(macro_f1, 4)

    return metrics


def _color_f1(f1: float) -> str:
    if f1 >= 0.85:
        return _green(f"{f1:.4f}")
    elif f1 >= 0.70:
        return _yellow(f"{f1:.4f}")
    return _red(f"{f1:.4f}")


def run_benchmark(wait: float = 0.0, dry_run: bool = False):
    rows = load_dataset(DATASET_PATH)
    if dry_run:
        rows = rows[:10]
        print(f"{_dim('[DRY RUN]')} Using first 10 records\n")

    total = len(rows)
    predictions = []
    start_time = time.time()

    # Pre-count ground truth
    gt_counts = defaultdict(int)
    for row in rows:
        gt_counts[row["true_label"]] += 1
    gt_line = "  ".join(f"{l}: {gt_counts[l]}" for l in LABELS)
    print(f"{_bold('Ground Truth:')} {gt_line}")
    print(f"{_bold('Target:')}       F1 >= 0.85 for all classes")
    print(f"{'=' * 70}\n")

    for i, row in enumerate(rows, 1):
        cid = row["citation_id"]
        citation = row["raw_citation"]
        true_label = row["true_label"]
        corruption_type = row.get("corruption_type", "")
        notes = row.get("notes", "")

        pred_label = "ERROR"
        confidence = 0.0
        source = ""
        reason = ""

        result = call_api(citation)
        if result is None:
            time.sleep(0.5)
            result = call_api(citation)

        if result is not None:
            pred_label = result.get("label", "ERROR")
            confidence = result.get("confidence", 0.0)
            source = result.get("source", "")
            reason = result.get("reason", "")

        correct = pred_label == true_label
        predictions.append({
            "citation_id": cid,
            "raw_citation": citation,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "corruption_type": corruption_type,
            "notes": notes,
            "source": source,
            "correct": correct,
        })

        # Live display
        elapsed = time.time() - start_time
        eta = (elapsed / i) * (total - i) if i > 0 else 0
        pct = i / total * 100

        bar_width = min(30, _terminal_width() - 50)
        filled = int(bar_width * i / total)
        bar = _green("=" * filled) + _dim("-" * (bar_width - filled))

        status = _green("OK") if correct else _red("XX")
        status_line = (
            f"\r[{i:>3}/{total}] [{bar}] {pct:5.1f}%  "
            f"[{status}] {true_label:>5} -> {pred_label:<17}  "
            f"ETA {eta:>5.0f}s"
        )

        # Truncate to terminal width
        tw = _terminal_width()
        if len(status_line) > tw:
            status_line = status_line[:tw - 3] + "..."

        sys.stdout.write(status_line)
        sys.stdout.flush()

        if i > 0 and i % 5 == 0:
            # Show running metrics pane
            metrics = compute_running_metrics(predictions)
            rows_display = []
            rows_display.append(f"\n\n  {'Class':<20} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'Target':<10}")
            rows_display.append(f"  {'-' * 58}")
            for label in LABELS:
                m = metrics[label]
                f1_display = _color_f1(m["f1"])
                target_status = _green("PASS") if m["f1"] >= 0.85 else _yellow("----") if m["f1"] >= 0.70 else _red("FAIL")
                rows_display.append(
                    f"  {label:<20} {f1_display:<16} {m['tp']:<6} {m['fp']:<6} {m['fn']:<6} {target_status:<10}"
                )
            rows_display.append(f"  {'-' * 58}")
            acc = metrics["accuracy"]
            mf1 = metrics["macro_f1"]
            rows_display.append(f"  {'Accuracy':<20} {acc:.4f}")
            rows_display.append(f"  {'Macro F1':<20} {_color_f1(mf1):<16}")
            rows_display.append("")
            sys.stdout.write("\n".join(rows_display))
            sys.stdout.flush()

        if wait > 0:
            time.sleep(wait)

    # Final metrics
    print(f"\n\n{'=' * 70}")
    print(_bold("FINAL RESULTS"))
    print(f"{'=' * 70}")
    final_metrics = compute_running_metrics(predictions)

    all_pass = True
    for label in LABELS:
        m = final_metrics[label]
        f1_display = _color_f1(m["f1"])
        status = _green("PASS") if m["f1"] >= 0.85 else _red("FAIL")
        if m["f1"] < 0.85:
            all_pass = False
        print(f"  {label:<20} F1={f1_display:<16} TP={m['tp']:<4} FP={m['fp']:<4} FN={m['fn']:<4} [{status}]")
    print(f"  {'-' * 60}")
    print(f"  {'Accuracy':<20} {final_metrics['accuracy']:.4f}")
    print(f"  {'Macro F1':<20} {_color_f1(final_metrics['macro_f1'])}")
    print(f"  {'Total samples':<20} {len(predictions)}")
    print(f"  {'Total time':<20} {time.time() - start_time:.1f}s")

    if all_pass:
        print(f"\n  {_green(_bold('ALL CLASSES PASSED F1 >= 0.85!'))}")
    else:
        print(f"\n  {_red(_bold('SOME CLASSES BELOW 0.85 THRESHOLD'))}")

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

    print(f"\n  Predictions written to {RESULTS_PATH}")

    return predictions, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run improved citation validation benchmark with live display"
    )
    parser.add_argument(
        "--wait", type=float, default=0.0,
        help="Delay in seconds between API calls (for rate limiting)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on the first 10 records only",
    )
    args = parser.parse_args()
    run_benchmark(wait=args.wait, dry_run=args.dry_run)
