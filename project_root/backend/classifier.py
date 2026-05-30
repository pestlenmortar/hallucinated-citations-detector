"""
Production inference wrapper for the trained classifier.

Drop-in replacement for heuristic_verify: takes a top candidate + parsed citation,
returns a label and confidence.

Handles both Pipeline (with scaler+smote built-in) and bare models
(with external scaler) for backward compatibility.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np

from backend.feature_engineering import extract_feature_vector, FEATURE_NAMES
from backend.thresholding import DEFAULT_THRESHOLDS, apply_thresholds

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "..", "models")

_label_map = {0: "VALID", 1: "PARTIALLY_VALID", 2: "HALLUCINATED"}

_model_cache: Any | None = None
_scaler_cache: Any | None = None
_model_feature_names: list[str] | None = None


def _load_model_and_scaler() -> tuple[Any, Any | None, list[str]]:
    global _model_cache, _scaler_cache, _model_feature_names
    if _model_cache is not None and _model_feature_names is not None:
        return _model_cache, _scaler_cache, _model_feature_names

    model_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    feature_path = os.path.join(MODEL_DIR, "n_features.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Run backend/train.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    used_features = FEATURE_NAMES
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            data = json.load(f)
            used_features = data.get("feature_names", FEATURE_NAMES)

    _model_cache = model
    _scaler_cache = scaler
    _model_feature_names = used_features

    return _model_cache, _scaler_cache, _model_feature_names


def _has_scaler_in_pipeline(model: Any) -> bool:
    """Check if model is an sklearn Pipeline that already includes a scaler step."""
    from sklearn.pipeline import Pipeline as SkPipeline
    from imblearn.pipeline import Pipeline as ImbPipeline
    if isinstance(model, (SkPipeline, ImbPipeline)):
        return any(step[0] == "scaler" for step in model.steps)
    return False


def classify(
    top_candidate: dict,
    parsed_citation: dict,
    exact_title_match: int = 0,
    exact_doi_match: int = 0,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """
    Classify a single citation using the trained model.

    Args:
        top_candidate: top match from fuse_candidates (or empty dict)
        parsed_citation: dict from parse_citation
        exact_title_match: 1 if normalized_title matched a DB record
        exact_doi_match: 1 if exact DB DOI matched
        thresholds: probability thresholds for VALID / PARTIALLY_VALID

    Returns:
        dict with keys: label, confidence, probabilities, reason
    """
    if thresholds is None:
        try:
            import json, os
            MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
            t_path = os.path.join(MODEL_DIR, "thresholds.json")
            with open(t_path) as tf:
                thresholds = json.load(tf)
        except Exception:
            thresholds = DEFAULT_THRESHOLDS

    if not top_candidate:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "probabilities": {},
            "reason": "No candidate provided",
        }

    model, scaler, model_features = _load_model_and_scaler()

    features: np.ndarray = extract_feature_vector(
        top_candidate, parsed_citation,
        exact_title_match=exact_title_match,
        exact_doi_match=exact_doi_match,
    )

    # Select only the features the model was trained on
    idx_map = [FEATURE_NAMES.index(f) for f in model_features if f in FEATURE_NAMES]
    features = features[idx_map]

    if _has_scaler_in_pipeline(model):
        X = features.reshape(1, -1)
    elif scaler is not None:
        X = scaler.transform(features.reshape(1, -1))
    else:
        X = features.reshape(1, -1)

    n_classes = 3
    try:
        probs_raw = model.predict_proba(X)[0]
    except Exception:
        try:
            probs_raw = model.decision_function(X)
            if probs_raw.ndim > 1:
                probs_raw = probs_raw[0]
            exp = np.exp(probs_raw - np.max(probs_raw))
            probs_raw = exp / exp.sum()
        except Exception:
            return {
                "label": "HALLUCINATED",
                "confidence": 0.0,
                "probabilities": {},
                "reason": "Model inference failed",
            }

    probs = np.zeros(n_classes, dtype=np.float64)
    for idx in range(min(len(probs_raw), n_classes)):
        probs[idx] = probs_raw[idx]
    if probs.sum() > 0:
        probs = probs / probs.sum()

    prob_dict: dict[str, float] = {}
    for idx, label in _label_map.items():
        if idx < len(probs):
            prob_dict[label] = round(float(probs[idx]), 4)

    label, confidence = apply_thresholds(probs, thresholds)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "probabilities": prob_dict,
        "reason": f"Classifier prediction (probabilities: {prob_dict})",
    }


def is_model_available() -> bool:
    """Check whether a trained model is available on disk."""
    model_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    return os.path.exists(model_path)


def invalidate_cache() -> None:
    """Clear cached model (forces reload on next classify call)."""
    global _model_cache, _scaler_cache
    _model_cache = None
    _scaler_cache = None
