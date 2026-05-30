#!/usr/bin/env python3
"""Bulk ingestion driver: runs multiple non-CS engineering queries across OpenAlex
to reach a target record count with broad domain coverage."""

import argparse
import os
import sqlite3
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import ingest_openalex

NON_CS_QUERIES = [
    "structural engineering",
    "transportation engineering",
    "geotechnical engineering",
    "thermodynamics",
    "fluid mechanics",
    "finite element analysis",
    "computational mechanics",
    "propulsion",
    "metallurgy",
    "power electronics",
    "power systems",
    "vlsi",
    "signal processing",
    "instrumentation engineering",
    "communication systems",
    "embedded systems",
    "battery systems",
    "renewable energy",
    "mechanical engineering",
    "electrical engineering",
    "civil engineering",
    "chemical engineering",
    "materials science",
    "aerospace engineering",
    "biomedical engineering",
    "nuclear engineering",
    "environmental engineering",
    "control systems",
    "robotics engineering",
    "manufacturing processes",
]


def count_records():
    conn = sqlite3.connect("papers.db")
    n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    return n


def run_source(source_name, ingest_fn, query, max_records):
    print(f"\n  [{source_name}] query='{query}' max={max_records}")
    try:
        ingest_fn(query, max_records=max_records)
    except Exception as e:
        print(f"  [{source_name}] FAILED: {e}")
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest non-CS engineering papers via OpenAlex free API"
    )
    parser.add_argument("--target", type=int, default=50000,
                        help="Target number of total records (default: 50000)")
    parser.add_argument("--source", choices=["openalex"],
                        default="openalex", help="Which source to use (default: openalex)")
    parser.add_argument("--queries", help="Comma-separated list of search queries "
                        "(overrides built-in list)")
    args = parser.parse_args()

    queries = args.queries.split(",") if args.queries else NON_CS_QUERIES
    queries = [q.strip() for q in queries if q.strip()]

    start_total = count_records()
    target = args.target
    needed = max(0, target - start_total)

    print(f"Starting ingestion. DB has {start_total} records. Target: {target}.")
    print(f"Need {needed} more. Running {len(queries)} queries.\n")

    total_inserted = 0
    query_index = 0
    current_total = start_total

    for query in queries:
        if target and current_total >= target:
            print(f"\nTarget of {target} reached. Stopping.")
            break

        query_index += 1
        remaining = target - current_total if target else None

        print(f"[{query_index}/{len(queries)}] '{query}'  (need {remaining or '∞'} more)")

        before = current_total

        run_source("openalex", ingest_openalex.ingest_all, query, remaining)

        current_total = count_records()
        added = current_total - before
        total_inserted += added
        print(f"  -> {added} new records (DB total: {current_total})\n")

        if target and current_total >= target:
            break

        time.sleep(1.0)

    final = current_total
    print(f"\n{'='*50}")
    print(f"Ingestion finished. DB total: {final} records (+{final - start_total})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
