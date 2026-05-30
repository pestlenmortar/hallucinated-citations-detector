#!/usr/bin/env python3
"""
snapshot_ingest_ml_ai.py
Downloads OpenAlex snapshot works partitions (sorted newest-first),
filters for ML/AI papers published >= 2014, inserts into papers.db.
Target: configurable. Temp space: one partition file at a time (~450 MB max).

Uses awscli (aws s3 cp --no-sign-request) for downloads and gzip streaming.
"""

import argparse
import gzip
import json
import os
import re
import signal
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

S3_PREFIX = "s3://openalex/data/works/"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
TMP_DIR = tempfile.mkdtemp(prefix="snapshot_works_")
SLEEP_SECONDS = 0.5
BATCH_SIZE = 1000

ML_AI_CONCEPT_IDS = {
    "C154945302",  # Artificial intelligence (L1)
    "C119857082",  # Machine learning (L1)
    "C204321447",  # Natural language processing (L1)
    "C31972630",   # Computer vision (L1)
    "C108583219",  # Deep learning (L2)
    "C50644808",   # Artificial neural network (L2)
    "C97541855",   # Reinforcement learning (L2)
    "C81363708",   # Convolutional neural network (L2)
    "C774472",     # Margin (machine learning) (L2)
    "C46686674",   # Boosting (machine learning) (L2)
    "C77967617",   # Active learning (machine learning) (L2)
    "C175202392",  # Time delay neural network (L3)
    "C2984842247", # Deep neural networks (L3)
    "C2944601119", # Residual neural network (L3)
    "C2988773926", # Generative adversarial network (L3)
    "C66322947",   # Transformer (L3)
    "C2778403875", # Adversarial machine learning (L3)
    "C115903097",  # Online machine learning (L3)
    "C11731999",   # Spiking neural network (L3)
    "C2779990667", # Hybrid neural network (L3)
    "C50100734",   # Rectifier (neural networks) (L5)
    "C33010914",   # Random neural network (L4)
    "C134342201",  # Probabilistic neural network (L4)
    "C86582703",   # Stochastic neural network (L4)
    "C47702885",   # Feedforward neural network (L3)
    "C2781121602", # Modular neural network (L4)
    "C812465",     # Cellular neural network (L3)
    "C2779094486", # Quantum machine learning (L4)
    "C176777502",  # Anticipation (artificial intelligence) (L2)
    "C44464901",   # Marketing and artificial intelligence (L3)
    "C207453521",  # Artificial intelligence, situated approach (L2)
    "C26205005",   # Symbolic artificial intelligence (L3)
    "C91557362",   # Music and artificial intelligence (L2)
    "C33766855",   # Physical neural network (L5)
    "C2779765954", # Confabulation (neural networks) (L3)
    "C2776145597", # Dropout (neural networks) (L2)
    "C2777946921", # Semantic analysis (machine learning) (L2)
    "C47330980",   # Transformer types (L5)
    "C22958824",   # Autotransformer (L5)
    "C30112582",   # Artificial Intelligence System (L2)
    "C177973122",  # Types of artificial neural networks (L2)
    "C147168706",  # Recurrent neural network (L3)
    "C118403218",  # Biological neural network (L2)
    "C78600465",   # Overfitting (L3)
}

ML_AI_KEYWORDS = [
    "machine learning", "artificial intelligence", "deep learning",
    "neural network", "reinforcement learning",
    "natural language processing", "computer vision",
    "convolutional neural", "recurrent neural", "graph neural",
    "generative adversarial", "generative model",
    "large language model", "language model",
    "diffusion model", "autoencoder",
    "representation learning", "transfer learning",
    "federated learning", "contrastive learning",
    "self-supervised", "semi-supervised", "few-shot", "zero-shot",
    "fine-tuning", "pretrain", "pre-train", "post-training",
    "attention mechanism", "self-attention", "multi-head attention",
    "encoder-decoder", "sequence-to-sequence",
    "transformer model", "transformers",
    "variational autoencoder", "normalizing flow",
    "neural architecture search", "knowledge distillation",
    "adversarial training", "adversarial attack",
    "continual learning", "meta-learning", "meta learning",
    "imitation learning", "in-context learning", "in context learning",
    "chain-of-thought", "prompt engineering",
    "retrieval augmented", "retrieval-augmented",
    "explainable ai", "interpretable machine",
    "multimodal", "multi-modal",
    "reinforcement learning from human",
    "word embedding", "sentence embedding",
    "gradient descent", "backpropagation",
    "model compression", "quantization",
]


def normalize_title(title):
    if not title:
        return None
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_authors(authorships):
    if not authorships:
        return None
    names = []
    for entry in authorships:
        author = entry.get("author", {})
        name = author.get("display_name")
        if name:
            names.append(name)
    return ", ".join(names) if names else None


def extract_venue(primary_location):
    if primary_location and primary_location.get("source"):
        return primary_location["source"].get("display_name")
    return None


def is_ml_ai_work(work):
    year = work.get("publication_year")
    if year is None or year < 2014:
        return False

    concepts = work.get("concepts", [])
    if not concepts:
        return False

    for concept in concepts:
        cid = concept.get("id", "").split("https://openalex.org/")[-1]
        if cid in ML_AI_CONCEPT_IDS:
            return True

        name = concept.get("display_name", "").lower()
        for kw in ML_AI_KEYWORDS:
            if kw in name:
                return True

    return False


def _cleanup():
    if os.path.isdir(TMP_DIR):
        try:
            shutil.rmtree(TMP_DIR)
        except Exception:
            pass


def _handle_signal(signum, frame):
    print(f"\nReceived signal {signum}, cleaning up...", flush=True)
    _cleanup()
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def list_partitions():
    result = subprocess.run(
        ["aws", "s3", "ls", S3_PREFIX, "--no-sign-request"],
        capture_output=True, text=True, timeout=30,
    )
    prefixes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[-1].endswith("/"):
            prefix = parts[-1].rstrip("/")
            if prefix.startswith("updated_date="):
                date_str = prefix.split("=")[-1]
                if date_str >= "2016-01-01":
                    prefixes.append(prefix)
    prefixes.sort(reverse=True)
    return prefixes


def process_partition(conn, prefix, needed):
    files_result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_PREFIX}{prefix}/", "--no-sign-request"],
        capture_output=True, text=True, timeout=30,
    )
    gz_files = []
    for line in files_result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".gz"):
            gz_files.append(parts[-1])

    partition_inserted = 0
    partition_scanned = 0

    print(f"  Downloading {len(gz_files)} file(s) from {prefix}...", flush=True)

    for gz_file in sorted(gz_files):
        if needed <= 0:
            break

        s3_path = f"{S3_PREFIX}{prefix}/{gz_file}"
        local_path = os.path.join(TMP_DIR, gz_file)

        subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path, "--no-sign-request", "--quiet"],
            check=True, timeout=600,
        )

        works_buffer = []
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    work = json.loads(line)
                except json.JSONDecodeError:
                    continue
                partition_scanned += 1

                if is_ml_ai_work(work):
                    works_buffer.append(work)

                    if len(works_buffer) >= BATCH_SIZE:
                        batch = _prepare_batch(works_buffer)
                        take = min(len(batch), max(0, needed - partition_inserted))
                        if take:
                            conn.executemany(
                                "INSERT OR IGNORE INTO papers (title, normalized_title, authors, year, venue, doi, abstract) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                batch[:take],
                            )
                            partition_inserted += take
                        works_buffer = []
                        if partition_inserted >= needed:
                            break

        if works_buffer:
            batch = _prepare_batch(works_buffer)
            take = min(len(batch), max(0, needed - partition_inserted))
            if take:
                conn.executemany(
                    "INSERT OR IGNORE INTO papers (title, normalized_title, authors, year, venue, doi, abstract) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch[:take],
                )
                partition_inserted += take

        os.remove(local_path)

        if partition_inserted >= needed:
            break

    return partition_inserted, partition_scanned


def _prepare_batch(works_batch):
    data = []
    for work in works_batch:
        title = work.get("title")
        normalized_title = normalize_title(title)
        authors = extract_authors(work.get("authorships"))
        year = work.get("publication_year")
        venue = extract_venue(work.get("primary_location"))
        doi = work.get("doi")
        if doi:
            doi_clean = doi.split("https://doi.org/")[-1]
        else:
            openalex_id = work.get("id", "")
            if openalex_id:
                doi_clean = "openalex:" + openalex_id.split("https://openalex.org/")[-1]
            else:
                continue
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        data.append((title, normalized_title, authors, year, venue, doi_clean, abstract))
    return data


def main():
    parser = argparse.ArgumentParser(description="Ingest ML/AI papers from OpenAlex snapshot")
    parser.add_argument("--target", type=int, default=1_000_000,
                        help="Target number of ML/AI papers to ingest (default: 1,000,000)")
    parser.add_argument("--skip-newest", type=int, default=0,
                        help="Skip the N newest partitions (already processed)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a specific partition (e.g., updated_date=2026-02-10)")
    args = parser.parse_args()

    print(f"Target: {args.target:,} ML/AI papers from 2014+")
    print(f"Temp dir: {TMP_DIR}")
    print()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())

        start_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        print(f"Starting DB size: {start_count:,} papers")
        print()

        partitions = list_partitions()

        if args.skip_newest:
            skipped = partitions[:args.skip_newest]
            partitions = partitions[args.skip_newest:]
            print(f"Skipping {len(skipped)} already-processed partitions (newest first)")

        if args.resume_from:
            while partitions and partitions[0] != args.resume_from:
                print(f"  Skipping {partitions[0]} (resuming from {args.resume_from})")
                partitions.pop(0)
            if not partitions:
                print(f"Partition {args.resume_from} not found!")
                conn.close()
                return

        print(f"Processing {len(partitions)} partitions (2016-2026)")
        print()

        inserted = 0
        total_scanned = 0
        checkpoint_path = os.path.join(os.path.dirname(__file__), ".ingest_checkpoint")

        for i, prefix in enumerate(partitions):
            remaining = args.target - inserted
            if remaining <= 0:
                break

            date_str = prefix.split("=")[-1]
            t0 = time.time()
            part_inserted = 0
            part_scanned = 0
            try:
                part_inserted, part_scanned = process_partition(conn, prefix, remaining)
            except subprocess.CalledProcessError as e:
                print(f"  [{i+1}/{len(partitions)}] {date_str} DOWNLOAD FAILED: {e} (will retry on resume)")
                with open(checkpoint_path, "w") as cp:
                    cp.write(prefix + "\n")
                continue
            except Exception as e:
                print(f"  [{i+1}/{len(partitions)}] {date_str} FAILED: {e}")
                continue
            elapsed = time.time() - t0

            inserted += part_inserted
            total_scanned += part_scanned
            conn.commit()

            actual_db = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            pct = inserted / args.target * 100 if args.target else 0

            # Check free disk space periodically
            disk_msg = ""
            if i % 10 == 0:
                stat = os.statvfs(os.path.dirname(DB_PATH))
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
                disk_msg = f" | disk free: {free_gb:.1f} GB"

            print(f"  [{i+1}/{len(partitions)}] {date_str}: "
                  f"{part_inserted:,} inserted, {part_scanned:,} scanned "
                  f"({elapsed:.0f}s) | total {inserted:,} ({pct:.1f}%) "
                  f"| DB: {actual_db:,}{disk_msg}")

            # Checkpoint after each partition for resume support
            with open(checkpoint_path, "w") as cp:
                cp.write(prefix + "\n")

            time.sleep(SLEEP_SECONDS)

        # Clean checkpoint on success
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    finally:
        try:
            conn.close()
        except Exception:
            pass
        _cleanup()

    print()
    print(f"{'='*60}")
    actual_final = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    sqlite3.connect(DB_PATH).close()
    print(f"Ingestion complete. Attempted {inserted:,} inserts.")
    print(f"DB total: {actual_final:,}")


if __name__ == "__main__":
    main()
