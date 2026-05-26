import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import DB_PATH, FAISS_INDEX_PATH
from backend.semantic_search import build_faiss_index


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from paper titles in the database"
    )
    parser.add_argument("--db", default=DB_PATH,
                        help=f"Path to SQLite database (default: {DB_PATH})")
    parser.add_argument("--index", default=FAISS_INDEX_PATH,
                        help=f"Output path for FAISS index (default: {FAISS_INDEX_PATH})")
    args = parser.parse_args()

    print(f"Database: {args.db}")
    print(f"Index output: {args.index}")
    print(f"Mapping output: {args.index}.mapping.json")
    print("Building FAISS index ...")
    build_faiss_index(args.db, args.index)
    print("Done.")


if __name__ == "__main__":
    main()
