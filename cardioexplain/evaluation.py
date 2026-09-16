"""Binary classification metrics and deterministic evaluation plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def choose_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a validation threshold that maximizes F1; ties favor 0.5 proximity."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = np.array([f1_score(targets, probabilities >= value, zero_division=0)
                       for value in thresholds])
    best = np.flatnonzero(scores == scores.max())
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def classification_metrics(targets: np.ndarray, probabilities: np.ndarray,
                           threshold: float) -> dict[str, float | list[list[int]]]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(targets)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(targets, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(targets, probabilities)) if both_classes else float("nan"),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "samples": int(len(targets)),
        "positive_samples": int(targets.sum()),
    }


def save_evaluation_plots(targets: np.ndarray, probabilities: np.ndarray,
                          threshold: float, output_dir: str | Path) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label",
           title="Cardiomegaly confusion matrix")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close(fig)

    if len(np.unique(targets)) == 2:
        fpr, tpr, _ = roc_curve(targets, probabilities)
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
        axes[1].plot(recall, precision)
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
        fig.tight_layout(); fig.savefig(output_dir / "roc_pr_curves.png", dpi=180); plt.close(fig)


def save_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
