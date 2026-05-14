"""Evaluation: accuracy, per-class metrics, confusion matrix, ROC curves."""
from __future__ import annotations

from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from .dataset import EMOTION_LABELS, NUM_CLASSES


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: str | torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_proba) where y_proba has shape (N, num_classes)."""
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    try:
        macro_auc = roc_auc_score(label_binarize(y_true, classes=list(range(NUM_CLASSES))), y_proba, average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = float("nan")
    return {
        "accuracy": float(acc),
        "macro_auc": float(macro_auc),
        "report": report,
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        data = cm_norm; fmt = ".2f"; title = "Confusion Matrix (row-normalised)"
    else:
        data = cm; fmt = "d"; title = "Confusion Matrix"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(EMOTION_LABELS):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        try:
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
