import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "results", "predictions.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

LABELS = ["VALID", "PARTIALLY_VALID", "HALLUCINATED"]


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["predicted_label"] != "ERROR"].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with predicted_label=ERROR")
    return df


def print_metrics(y_true: pd.Series, y_pred: pd.Series):
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}\n")

    report = classification_report(
        y_true, y_pred, labels=LABELS, digits=4, zero_division=0
    )
    print(report)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for i, label in enumerate(LABELS):
        print(f"{label:<20} {prec[i]:<12.4f} {rec[i]:<12.4f} {f1[i]:<12.4f}")
    print("-" * 60)
    print(f"{'Macro F1':<20} {'':<12} {'':<12} {macro_f1:<12.4f}")
    print("=" * 60)

    return acc, macro_f1


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_f1_barchart(y_true: pd.Series, y_pred: pd.Series, path: str):
    f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2e7d32", "#e65100", "#c62828"]
    bars = ax.bar(LABELS, f1, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confidence_distribution(df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"VALID": "#2e7d32", "PARTIALLY_VALID": "#e65100", "HALLUCINATED": "#c62828"}

    for ax, label in zip(axes, LABELS):
        subset = df[df["true_label"] == label]["confidence"]
        ax.hist(subset, bins=20, range=(0, 1), color=colors[label],
                edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Count" if label == "VALID" else "", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Confidence Score Distribution by True Label", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_misclassification_by_corruption(df: pd.DataFrame, path: str):
    partial = df[df["true_label"] == "PARTIALLY_VALID"].copy()
    partial["misclassified"] = partial["true_label"] != partial["predicted_label"]

    grouped = (
        partial.groupby("corruption_type")["misclassified"]
        .agg(["count", "sum"])
        .reset_index()
    )
    grouped.columns = ["corruption_type", "total", "misclassified"]

    grouped = grouped.sort_values("misclassified", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d32f2f" if m > 0 else "#388e3c" for m in grouped["misclassified"]]
    bars = ax.bar(grouped["corruption_type"], grouped["misclassified"],
                  color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, grouped["misclassified"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(int(val)), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Corruption Type", fontsize=11)
    ax.set_ylabel("Misclassified Count", fontsize=11)
    ax.set_title("Misclassification by Corruption Type (PARTIALLY_VALID only)", fontsize=12)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped["corruption_type"], rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_predictions(PREDICTIONS_PATH)
    if df.empty:
        print("No valid predictions to evaluate.")
        return

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print_metrics(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred,
                          os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_f1_barchart(y_true, y_pred,
                     os.path.join(PLOTS_DIR, "f1_barchart.png"))
    plot_confidence_distribution(df,
                                 os.path.join(PLOTS_DIR, "confidence_distribution.png"))
    plot_misclassification_by_corruption(df,
                                          os.path.join(PLOTS_DIR, "misclassification_by_corruption.png"))

    print("\nAll plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
