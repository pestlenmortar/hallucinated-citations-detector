import os
import sqlite3
import sys
import types
from unittest.mock import MagicMock, patch

# --- mock pydantic before any backend imports ---
_mock_pd = types.ModuleType("pydantic")

class _MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

_mock_pd.BaseModel = _MockBaseModel
sys.modules["pydantic"] = _mock_pd

# --- set up the backend package ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

_backend = types.ModuleType("backend")
_backend.__path__ = [os.path.join(_project_root, "backend")]
sys.modules["backend"] = _backend

# --- backend.models ---
_models_src = open(os.path.join(_project_root, "backend", "models.py")).read()
_models_ns = {"__name__": "backend.models", "__package__": "backend", "__builtins__": __builtins__}
exec(_models_src, _models_ns)
_models_mod = types.ModuleType("backend.models")
_models_mod.__dict__.update(_models_ns)
sys.modules["backend.models"] = _models_mod

# --- backend.normalization ---
import re  # noqa: E402
import unicodedata  # noqa: E402

_norm_ns: dict = {
    "__name__": "backend.normalization",
    "__package__": "backend",
    "__builtins__": __builtins__,
    "re": __import__("re"),
    "unicodedata": __import__("unicodedata"),
}
with open(os.path.join(_project_root, "backend", "normalization.py")) as _f:
    exec(_f.read(), _norm_ns)
_norm_mod = types.ModuleType("backend.normalization")
_norm_mod.__dict__.update(_norm_ns)
sys.modules["backend.normalization"] = _norm_mod

# --- load fusion module ---
_fusion_ns: dict = {
    "__name__": "backend.fusion",
    "__package__": "backend",
    "__builtins__": __builtins__,
}
with open(os.path.join(_project_root, "backend", "fusion.py")) as _f:
    exec(_f.read(), _fusion_ns)

fuse_candidates = _fusion_ns["fuse_candidates"]


# ---------------------------------------------------------------------------
# Mock rows: (paper_id, title, authors, year, venue, doi, abstract)
MOCK_DB_ROWS = [
    (1, "Machine Learning", "Smith, J.", 2020, "Journal of AI", "10.1234/ml", "An introduction to ML concepts and algorithms."),
    (2, "Deep Learning", "Doe, J.", 2021, "Neural Computing", "10.1234/dl", "A survey of deep neural network approaches."),
    (3, "Reinforcement Learning", "Lee, K.", 2019, "IEEE Trans", "10.1234/rl", "Policy gradient methods for control tasks."),
]


def _mock_connection_factory(rows):
    def mock_connect(path):
        conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        conn.execute.return_value = mock_cursor
        return conn
    return mock_connect


def test_fuse_candidates_top_is_highest_final_score():
    fuzzy_results = [
        {"paper_id": 1, "title": "ML", "score": 85.0},
        {"paper_id": 2, "title": "DL", "score": 70.0},
        {"paper_id": 3, "title": "RL", "score": 60.0},
    ]
    semantic_results = [
        {"paper_id": 2, "score": 90.0},
        {"paper_id": 1, "score": 80.0},
    ]
    parsed_citation = {"authors": "Smith, J.", "year": 2020, "venue": "Journal"}

    with patch.object(sqlite3, "connect", _mock_connection_factory(MOCK_DB_ROWS)):
        fused = fuse_candidates(fuzzy_results, semantic_results, parsed_citation, ":memory:")

    assert len(fused) > 0
    # Paper 1 should win: fuzzy 85, semantic 80, strong metadata match
    top = fused[0]
    assert top["paper_id"] == 1
    assert top["final_score"] > 0

    # Verify sorted descending
    for i in range(len(fused) - 1):
        assert fused[i]["final_score"] >= fused[i + 1]["final_score"]


def test_fuse_candidates_includes_all_fields():
    fuzzy_results = [{"paper_id": 1, "title": "X", "score": 95.0}]
    semantic_results = []
    parsed_citation = {"authors": "", "year": None, "venue": ""}

    with patch.object(sqlite3, "connect", _mock_connection_factory(MOCK_DB_ROWS)):
        fused = fuse_candidates(fuzzy_results, semantic_results, parsed_citation, ":memory:")

    assert len(fused) >= 1
    top = fused[0]
    for key in ("paper_id", "title", "authors", "year", "venue",
                "fuzzy_score", "semantic_score", "metadata_score",
                "author_similarity", "year_similarity", "venue_similarity",
                "doi_similarity", "abstract_similarity", "final_score"):
        assert key in top, f"missing key: {key}"


def test_fuse_candidates_empty_inputs():
    with patch.object(sqlite3, "connect", _mock_connection_factory(MOCK_DB_ROWS)):
        fused = fuse_candidates([], [], {"authors": "", "year": None, "venue": ""}, ":memory:")
    assert fused == []
