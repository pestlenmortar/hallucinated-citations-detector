import os
import sys
import types

# --- mock pydantic before any backend imports ---
_mock_pd = types.ModuleType("pydantic")

class _MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

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
    "re": re,
    "unicodedata": unicodedata,
}
with open(os.path.join(_project_root, "backend", "normalization.py")) as _f:
    exec(_f.read(), _norm_ns)

_norm_mod = types.ModuleType("backend.normalization")
_norm_mod.__dict__.update(_norm_ns)
sys.modules["backend.normalization"] = _norm_mod

# --- backend.parser ---
_parser_ns: dict = {
    "__name__": "backend.parser",
    "__package__": "backend",
    "__builtins__": __builtins__,
}
with open(os.path.join(_project_root, "backend", "parser.py")) as _f:
    exec(_f.read(), _parser_ns)

parse_citation = _parser_ns["parse_citation"]


# ---------------------------------------------------------------------------


def test_parse_apa_citation():
    raw = "Smith, J., & Doe, J. (2020). Machine learning is great. Journal of AI, 10(2), 100-110. https://doi.org/10.1234/jai.2020.001"
    pc = parse_citation(raw).model_dump()
    assert pc["year"] == 2020
    assert "machine learning is great" == pc["title"]
    assert "Smith, J" in pc["authors"]


def test_parse_ieee_citation():
    raw = 'J. Smith and J. Doe, "Deep Learning Methods," IEEE Transactions, vol. 15, pp. 200-210, 2019, doi:10.5678/ieee.2019.002.'
    pc = parse_citation(raw).model_dump()
    assert pc["year"] == 2019
    assert "deep learning methods" == pc["title"]
    assert "Smith" in pc["authors"]


def test_parse_bare_citation():
    raw = "A standalone paper title without authors (2018). Some proceedings venue, 42."
    pc = parse_citation(raw).model_dump()
    assert pc["year"] == 2018
    assert "a standalone paper title without authors" == pc["title"]
    assert pc["authors"] == ""
