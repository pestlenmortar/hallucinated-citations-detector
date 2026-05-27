import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "test_citations.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "predictions.csv")
API_URL = "http://localhost:8000/validate"
TIMEOUT = 30


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


def run_benchmark(dry_run: bool = False):
    rows = load_dataset(DATASET_PATH)
    if dry_run:
        rows = rows[:10]
        print(f"[DRY RUN] Using first {len(rows)} records")

    total = len(rows)
    predictions = []

    for i, row in enumerate(rows, 1):
        cid = row["citation_id"]
        citation = row["raw_citation"]
        true_label = row["true_label"]
        corruption_type = row.get("corruption_type", "")
        notes = row.get("notes", "")

        pred_label = "ERROR"
        confidence = 0.0

        result = call_api(citation)
        if result is None:
            time.sleep(0.5)
            result = call_api(citation)

        if result is not None:
            pred_label = result.get("label", "ERROR")
            confidence = result.get("confidence", 0.0)

        predictions.append({
            "citation_id": cid,
            "raw_citation": citation,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "corruption_type": corruption_type,
            "notes": notes,
        })

        if i % 10 == 0:
            print(f"[{i}/{total}] Processed {i} citations")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = [
        "citation_id", "raw_citation", "true_label", "predicted_label",
        "confidence", "corruption_type", "notes",
    ]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Wrote {len(predictions)} predictions to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run citation validation benchmark against the API"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on the first 10 records only",
    )
    args = parser.parse_args()
    run_benchmark(dry_run=args.dry_run)
