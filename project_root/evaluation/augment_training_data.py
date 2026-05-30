"""
Augment training dataset: add derived features from existing 15 features
(interactions, nonlinear transforms, etc.) without re-running the pipeline.

Writes augmented CSV to evaluation/results/training_dataset_augmented.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.feature_engineering import FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "results", "training_dataset_full.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "training_dataset_augmented.csv")


def main():
    rows = []
    with open(INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    fieldnames = ["citation_id", "true_label", "corruption_type", "notes"] + FEATURE_NAMES
    total = len(rows)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            out = {
                "citation_id": row["citation_id"],
                "true_label": row["true_label"],
                "corruption_type": row.get("corruption_type", ""),
                "notes": row.get("notes", ""),
            }

            # Copy existing 15 features
            for name in FEATURE_NAMES[:15]:
                out[name] = row.get(name, 0)

            # Compute derived features from existing ones
            ts = float(out["title_similarity"])
            ao = float(out["author_overlap"])
            dm = float(out["doi_match"])
            abst = float(out["abstract_similarity"])
            sem = float(out["semantic_similarity"])
            vs = float(out["venue_similarity"])  
            ys = float(out["year_similarity"])
            fus = float(out["fusion_final_score"])
            ms = float(out["metadata_score"])
            etm = float(out["exact_title_match"])
            edm = float(out["exact_doi_match"])
            tld = float(out["title_length_difference"])
            acd = float(out["author_count_difference"])
            nyg = float(out["normalized_year_gap"])
            tdm = float(out["top_doi_match"])

            # New features
            out["title_char_similarity"] = ts  # proxy: use token similarity
            acr = 1.0 / (1.0 + acd) if acd >= 0 else 0.5  # author count ratio proxy
            out["author_count_ratio"] = round(acr, 4)
            out["title_word_count_diff_ratio"] = tld  # proxy: use length diff
            out["title_author_interaction"] = round(ts * ao, 4)
            out["title_year_interaction"] = round(ts * ys, 4)
            out["semantic_title_interaction"] = round(sem * ts, 4)
            out["doi_presence_penalty"] = 1.0 if (edm == 0 and tdm == 0 and dm == 0) else 0.0
            out["first_author_match"] = 0.0  # cannot compute from existing
            out["venue_similarity_sq"] = round(vs * vs, 4)
            out["author_overlap_sq"] = round(ao * ao, 4)
            f_ratio = ms * 100 / max(fus, 0.01) if fus > 0 else 0.0
            out["fusion_metadata_ratio"] = round(f_ratio, 4)
            out["year_match_binary"] = 1.0 if ys >= 1.0 else 0.0
            sc = min(ts, ao, max(sem, 0.1)) if fus > 30 else 0.0
            out["score_certainty"] = round(sc, 4)
            out["semantic_boost"] = round(max(0.0, sem - ts), 4)
            out["title_final_interaction"] = round(ts * fus / 100.0, 4)

            writer.writerow(out)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{total}")

    print(f"Wrote {total} samples with {len(FEATURE_NAMES)} features to {OUTPUT_PATH}")

    # Verify
    with open(OUTPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        feature_cols = [c for c in cols if c in FEATURE_NAMES]
        print(f"  Feature columns in output: {len(feature_cols)}")
        sample = next(reader)
        print(f"  Sample: VALID={sample.get('VALID', 'N/A')}, title_similarity={sample.get('title_similarity', 'N/A')}, title_char_similarity={sample.get('title_char_similarity', 'N/A')}")


if __name__ == "__main__":
    main()
