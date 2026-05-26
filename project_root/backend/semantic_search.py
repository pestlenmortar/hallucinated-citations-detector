import json
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_model():
    _get_model()


def _make_content(row):
    title = row[1] or ""
    abstract = row[2] or ""
    if abstract:
        return title + ". " + abstract
    return title


def build_faiss_index(db_path: str, index_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE title IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    paper_ids = [row[0] for row in rows]
    contents = [_make_content(row) for row in rows]

    model = _get_model()
    embeddings = model.encode(contents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    mapping = {i: pid for i, pid in enumerate(paper_ids)}
    with open(index_path + ".mapping.json", "w") as f:
        json.dump(mapping, f)


def semantic_search(query_title: str, index_path: str, k: int = 30) -> list[dict]:
    index = faiss.read_index(index_path)

    with open(index_path + ".mapping.json", "r") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}

    model = _get_model()
    query_vec = model.encode([query_title])
    query_vec = np.array(query_vec).astype("float32")

    k = min(k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {"paper_id": mapping[idx], "score": round(float(dist), 4)}
        )

    return results
