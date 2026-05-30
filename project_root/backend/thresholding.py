"""
Probability threshold tuning for the citation classifier.

Multi-objective grid search over threshold pairs to maximize
per-class F1 scores, with HALLUCINATED->VALID leakage hard-capped at 0.

Also supports calibration via Platt scaling / isotonic regression.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

DEFAULT_THRESHOLDS: dict[str, float] = {
    "valid_threshold": 0.55,
    "partial_threshold": 0.55,
}

LABELS: list[str] = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]
LABEL_INDEX: dict[str, int] = {"VALID": 0, "PARTIALLY_VALID": 1, "HALLUCINATED": 2}


def apply_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, float]:
    """
    Convert predicted probabilities into a label using tuned thresholds.

    Decision logic:
      1. If P(HALLUCINATED) is highest AND >= 0.50  -> HALLUCINATED
      2. If P(VALID) >= valid_threshold             -> VALID
      3. If P(VALID)+P(PARTIALLY_VALID) >= partial_threshold  -> PARTIALLY_VALID
      4. Otherwise                                  -> HALLUCINATED

    Args:
        probabilities: shape (3,) -- P(VALID), P(PARTIALLY_VALID), P(HALLUCINATED)
        thresholds: dict with 'valid_threshold' and 'partial_threshold'

    Returns:
        (label_str, confidence_float)
    """
    t = thresholds or DEFAULT_THRESHOLDS
    vt: float = t.get("valid_threshold", 0.55)
    pt: float = t.get("partial_threshold", 0.50)

    p_valid: float = float(probabilities[0]) if len(probabilities) > 0 else 0.0
    p_partial: float = float(probabilities[1]) if len(probabilities) > 1 else 0.0
    p_hall: float = float(probabilities[2]) if len(probabilities) > 2 else 0.0

    if p_hall >= 0.50 and p_hall > p_valid and p_hall > p_partial:
        return "HALLUCINATED", p_hall

    if p_valid >= vt:
        return "VALID", p_valid
    elif p_valid + p_partial >= pt:
        return "PARTIALLY_VALID", max(p_partial, 0.0)
    else:
        return "HALLUCINATED", p_hall


def _count_errors(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    vt: float,
    pt: float,
) -> dict[str, int]:
    """Count the confusion matrix entries given thresholds."""
    counts = {f"{t}_{p}": 0 for t in LABELS for p in LABELS}
    for i in range(len(probabilities)):
        label, _ = apply_thresholds(probabilities[i], {"valid_threshold": vt, "partial_threshold": pt})
        true_label = LABELS[y_true[i]]
        counts[f"{true_label}_{label}"] += 1
    return counts


def _f1_from_counts(counts: dict, label: str) -> float:
    tp = counts.get(f"{label}_{label}", 0)
    fp = sum(counts.get(f"{l}_{label}", 0) for l in LABELS if l != label)
    fn = sum(counts.get(f"{label}_{l}", 0) for l in LABELS if l != label)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _hall_to_valid_count(probabilities, y_true, vt, pt):
    counts = _count_errors(probabilities, y_true, vt, pt)
    return counts.get("HALLUCINATED_VALID", 0)


def grid_search_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> dict[str, float]:
    """
    Grid-search over valid_threshold and partial_threshold to maximise
    minimum per-class F1 while keeping HALLUCINATED -> VALID = 0.

    Args:
        probabilities: (N, 3) predicted probabilities
        y_true: (N,) true label indices
        y_pred_baseline: (N,) default (argmax) predictions for reference

    Returns:
        Best thresholds dict.
    """
    best_vt, best_pt = 0.45, 0.30
    best_score: float = -1.0

    for vt in np.arange(0.15, 0.90, 0.05):
        for pt in np.arange(0.15, 0.75, 0.05):
            if pt > vt:
                continue

            hv_count = _hall_to_valid_count(probabilities, y_true, vt, pt)
            if hv_count > 0:
                continue

            counts = _count_errors(probabilities, y_true, vt, pt)
            f1_valid = _f1_from_counts(counts, "VALID")
            f1_partial = _f1_from_counts(counts, "PARTIALLY_VALID")
            f1_hall = _f1_from_counts(counts, "HALLUCINATED")

            min_f1 = min(f1_valid, f1_partial, f1_hall)
            macro_f1 = (f1_valid + f1_partial + f1_hall) / 3.0

            score = min_f1 * 0.4 + macro_f1 * 0.6

            if score > best_score:
                best_score = score
                best_vt = float(vt)
                best_pt = float(pt)

    return {
        "valid_threshold": round(best_vt, 2),
        "partial_threshold": round(best_pt, 2),
    }
