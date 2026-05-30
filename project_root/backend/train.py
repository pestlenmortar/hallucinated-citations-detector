"""
Training pipeline for the citation hallucination classifier.

Trains multiple models (LogisticRegression, XGBoost, LightGBM, SVM) with
RandomizedSearchCV, SMOTE oversampling, and ensemble voting on retrieval-
computed features, optimising for macro F1.

Uses stratified 5-fold cross-validation as the primary evaluation (with only
200 labeled samples, a held-out test set would be too small for reliable
metrics).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from backend.feature_engineering import FEATURE_NAMES

RANDOM_SEED: int = 42
LABELS: list[str] = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]
LABEL_TO_ID: dict[str, int] = {l: i for i, l in enumerate(LABELS)}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "..", "models")
CSV_DEFAULT: str = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "results", "training_dataset_full.csv",
)


def _label_to_id(label: str) -> int:
    return LABEL_TO_ID.get(label.strip(), -1)


def load_all_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load full dataset without splitting. Returns X, y, ids, and list of feature names used."""
    df = pd.read_csv(csv_path)
    df = df[df["true_label"].isin(LABELS)].copy()

    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    if not available_features:
        raise ValueError(f"No feature columns found in {csv_path}. Expected at least one of {FEATURE_NAMES[:5]}...")

    print(f"  Found {len(available_features)}/{len(FEATURE_NAMES)} features in dataset")

    X = np.array([
        [float(row.get(name, 0.0)) for name in available_features]
        for _, row in df.iterrows()
    ], dtype=np.float64)
    y = np.array([_label_to_id(row["true_label"]) for _, row in df.iterrows()], dtype=np.int32)
    ids = df["citation_id"].values
    return X, y, ids, available_features


def _train_estimator(X_tr: np.ndarray, y_tr: np.ndarray, estimator: Any) -> Any:
    """Train with SMOTE oversampling."""
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=min(3, min(np.bincount(y_tr)) - 1))),
        ("clf", clone(estimator)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


_XGB_PARAM_GRID: dict[str, Any] = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
}

_LGBM_PARAM_GRID: dict[str, Any] = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 5, -1],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__num_leaves": [15, 31],
}

_LOGISTIC_PARAM_GRID: dict[str, Any] = {
    "clf__C": [0.1, 0.5, 1.0, 5.0],
    "clf__solver": ["lbfgs", "newton-cholesky"],
}

_SVM_PARAM_GRID: dict[str, Any] = {
    "clf__C": [0.5, 1.0, 5.0],
    "clf__gamma": ["scale", "auto"],
    "clf__kernel": ["rbf"],
    "clf__class_weight": ["balanced"],
}

MODEL_FACTORY: dict[str, Any] = {
    "logistic": lambda: LogisticRegression(
        max_iter=5000, random_state=RANDOM_SEED, class_weight="balanced",
    ),
    "xgboost": lambda: XGBClassifier(
        random_state=RANDOM_SEED, eval_metric="mlogloss", verbosity=0,
    ),
    "lightgbm": lambda: LGBMClassifier(
        random_state=RANDOM_SEED, verbose=-1, class_weight="balanced",
    ),
    "svm": lambda: SVC(
        probability=True, random_state=RANDOM_SEED, class_weight="balanced",
    ),
}

MODEL_PARAM_GRIDS: dict[str, dict] = {
    "logistic": _LOGISTIC_PARAM_GRID,
    "xgboost": _XGB_PARAM_GRID,
    "lightgbm": _LGBM_PARAM_GRID,
    "svm": _SVM_PARAM_GRID,
}

DISPLAY_NAMES: dict[str, str] = {
    "logistic": "LogisticRegression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "svm": "SVM",
    "ensemble": "Ensemble",
}


def evaluate_model(
    model: Any, X: np.ndarray, y: np.ndarray, name: str = "model",
) -> dict[str, Any]:
    """Evaluate a fitted model. Returns metrics dict."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, labels=range(len(LABELS)), average="macro", zero_division=0)
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        y, y_pred, labels=range(len(LABELS)), zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=range(len(LABELS)))

    per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1_per[i]), 4),
        }

    return {
        "name": name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y)),
    }


def cross_validate_model(
    model_type: str, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
) -> dict[str, Any]:
    """Stratified k-fold CV (no per-fold tuning for speed). Returns per-fold metrics + aggregates."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict[str, Any]] = []
    hall_to_valid_counts: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = _train_estimator(X_tr, y_tr, MODEL_FACTORY[model_type]())
        metrics = evaluate_model(model, X_va, y_va, name=f"fold{fold}")
        fold_metrics.append(metrics)

        cm = np.array(metrics["confusion_matrix"])
        hall_to_valid_counts.append(int(cm[2, 0]) if cm.shape[0] > 2 else 0)

    mf1_scores = [m["macro_f1"] for m in fold_metrics]
    acc_scores = [m["accuracy"] for m in fold_metrics]

    return {
        "model_type": model_type,
        "n_folds": n_splits,
        "n_samples": int(len(X)),
        "fold_metrics": fold_metrics,
        "macro_f1_mean": round(float(np.mean(mf1_scores)), 4),
        "macro_f1_std": round(float(np.std(mf1_scores)), 4),
        "accuracy_mean": round(float(np.mean(acc_scores)), 4),
        "hall_to_valid_mean": round(float(np.mean(hall_to_valid_counts) if hall_to_valid_counts else 0), 2),
        "hall_to_valid_folds": hall_to_valid_counts,
    }


def train_all(csv_path: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
    """
    Full training pipeline:

    1. Load all data (no held-out split)
    2. 5-fold stratified CV with hyperparameter tuning for model selection
    3. Train best model + ensemble on ALL data
    4. Save model + scaler + metrics
    """
    path = csv_path or CSV_DEFAULT
    mdir = model_dir or MODEL_DIR
    os.makedirs(mdir, exist_ok=True)

    X_raw, y, ids, used_features = load_all_data(path)
    n_feat = X_raw.shape[1]
    print(f"Loaded {len(X_raw)} samples x {n_feat} features")
    print(f"Class distribution: {dict(zip(LABELS, np.bincount(y)))}")

    results: dict[str, Any] = {
        "n_samples": int(len(y)),
        "feature_names": used_features,
        "class_distribution": {LABELS[i]: int(c) for i, c in enumerate(np.bincount(y))},
        "models": {},
    }
    best_cv_mf1: float = -1.0
    best_type: str = "logistic"
    best_model_cv_info: dict | None = None

    model_types = ["logistic", "xgboost", "lightgbm", "svm"]

    for model_type in model_types:
        print(f"\n  Tuning/evaluating {DISPLAY_NAMES[model_type]} ...")
        cv = cross_validate_model(model_type, X_raw, y, n_splits=5)
        results["models"][model_type] = cv
        dname = DISPLAY_NAMES[model_type]
        print(
            f"  {dname:>22s}  "
            f"CV mF1 = {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f}  "
            f"acc = {cv['accuracy_mean']:.4f}  "
            f"H->V = {cv['hall_to_valid_mean']:.1f}"
        )
        if cv["macro_f1_mean"] > best_cv_mf1:
            best_cv_mf1 = cv["macro_f1_mean"]
            best_type = model_type
            best_model_cv_info = cv

    # Fit best single model on all data
    print(f"\n  Training best model ({DISPLAY_NAMES[best_type]}) on all {len(y)} samples ...")
    dname = DISPLAY_NAMES[best_type]

    # Do final RandomizedSearchCV on full data for best model
    final_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            random_state=RANDOM_SEED,
            k_neighbors=min(3, min(np.bincount(y)) - 1),
        )),
        ("clf", clone(MODEL_FACTORY[best_type]())),
    ])
    if best_type in MODEL_PARAM_GRIDS:
        n_iter = min(15, np.prod([len(v) for v in MODEL_PARAM_GRIDS[best_type].values()]))
        search = RandomizedSearchCV(
            final_pipe,
            MODEL_PARAM_GRIDS[best_type],
            n_iter=n_iter,
            cv=5,
            scoring="f1_macro",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        search.fit(X_raw, y)
        best_model = search.best_estimator_
        print(f"  Best params: {search.best_params_}")
    else:
        best_model = _train_estimator(X_raw, y, MODEL_FACTORY[best_type]())

    # Also build a simpler ensemble of the top 2 models besides the best one
    print(f"\n  Training ensemble classifier ...")
    cv_results_sorted = sorted(
        [(k, v) for k, v in results["models"].items()],
        key=lambda x: x[1]["macro_f1_mean"],
        reverse=True,
    )
    top3_types = [t for t, _ in cv_results_sorted[:3]]
    print(f"  Top 3 models for ensemble: {', '.join(DISPLAY_NAMES[t] for t in top3_types)}")

    ensemble_estimators = []
    for mtype in top3_types:
        epipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(
                random_state=RANDOM_SEED,
                k_neighbors=min(3, min(np.bincount(y)) - 1),
            )),
            ("clf", clone(MODEL_FACTORY[mtype]())),
        ])
        if mtype in MODEL_PARAM_GRIDS:
            n_iter = min(8, np.prod([len(v) for v in MODEL_PARAM_GRIDS[mtype].values()]))
            esearch = RandomizedSearchCV(
                epipe,
                MODEL_PARAM_GRIDS[mtype],
                n_iter=n_iter,
                cv=3,
                scoring="f1_macro",
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
            esearch.fit(X_raw, y)
            ensemble_estimators.append((mtype, esearch.best_estimator_))
        else:
            ensemble_estimators.append((mtype, _train_estimator(X_raw, y, MODEL_FACTORY[mtype]())))

    voting_clf = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
        weights=[1.0] * len(ensemble_estimators),
    )
    voting_clf.fit(X_raw, y)

    # Evaluate best single and ensemble on full data
    best_eval = evaluate_model(best_model, X_raw, y, name="full_data_best")
    print(f"  {dname} full-data macro F1: {best_eval['macro_f1']:.4f}")

    ensemble_eval = evaluate_model(voting_clf, X_raw, y, name="full_data_ensemble")
    print(f"  Ensemble full-data macro F1: {ensemble_eval['macro_f1']:.4f}")

    # Use ensemble if it's better, otherwise best single model
    use_ensemble = ensemble_eval["macro_f1"] > best_eval["macro_f1"]
    final_model = voting_clf if use_ensemble else best_model
    final_eval = ensemble_eval if use_ensemble else best_eval
    final_type = "ensemble" if use_ensemble else best_type

    print(f"\n  Saving {DISPLAY_NAMES[final_type]} as best_classifier.pkl ...")

    model_path = os.path.join(mdir, "best_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Save n_features so inference code knows how many features to use
    feature_count_path = os.path.join(mdir, "n_features.json")
    with open(feature_count_path, "w") as f:
        json.dump({"n_features": n_feat, "feature_names": used_features}, f)

    # Also save standalone scaler (fitted on raw data) for backward compat
    standalone_scaler = StandardScaler().fit(X_raw)
    scaler_path = os.path.join(mdir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(standalone_scaler, f)

    results["best_model"] = {
        "type": final_type,
        "cv_macro_f1": best_cv_mf1,
        "full_data_macro_f1": final_eval["macro_f1"],
        "model_path": model_path,
        "scaler_path": scaler_path,
        "ensemble": use_ensemble,
        "ensemble_models": top3_types if use_ensemble else [],
    }
    results["full_data_metrics"] = final_eval

    # Save metrics JSON
    metrics_path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation", "results", "train_metrics.json",
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Print final per-class metrics
    print("\n  Final Model Per-Class Metrics:")
    for label in LABELS:
        m = final_eval["per_class"].get(label, {})
        print(f"    {label:<20} P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  F1={m.get('f1',0):.4f}")

    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    train_all(os.path.abspath(path))
