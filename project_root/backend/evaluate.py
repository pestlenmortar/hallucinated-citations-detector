"""
Evaluation utilities: generate classification report, confusion matrix plot,
per-class F1 bar chart, and export metrics as JSON.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS: list[str] = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]

OUTPUT_DIR: str = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "results"
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(LABELS, fontsize=10)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > (cm.max() / 2) else "black",
                fontsize=11,
            )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "confusion_matrix_clf.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_f1_bar_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
):
    f1 = f1_score(y_true, y_pred, labels=range(len(LABELS)), average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2e7d32", "#e65100", "#c62828"]
    bars = ax.bar(LABELS, f1, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "f1_scores_clf.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | None = None,
) -> dict:
    """Generate comprehensive evaluation, plots, and metrics JSON."""
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=range(len(LABELS)), average="macro", zero_division=0,
    )
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))

    per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1_per[i]), 4),
        }

    report: dict = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "labels": LABELS,
    }

    # Save JSON
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to {json_path}")

    # Generate plots
    plot_confusion_matrix(y_true, y_pred,
                          os.path.join(out_dir, "confusion_matrix_clf.png"))
    plot_f1_bar_chart(y_true, y_pred,
                      os.path.join(out_dir, "f1_scores_clf.png"))

    return report
