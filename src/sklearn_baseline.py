"""Scikit-Learn baseline (logistic regression on raw pixels).
Included to satisfy the 'support multiple frameworks' requirement and to give
the experimental report a non-deep-learning point of comparison.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .dataset import build_dataloaders


def _flatten_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy().reshape(x.size(0), -1))
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def train_sklearn_baseline(data_dir: str, save_path: str = "checkpoints/sklearn_logreg.pkl",
                           max_iter: int = 200, c: float = 1.0) -> dict:
    train_loader, val_loader, test_loader = build_dataloaders(data_dir, batch_size=256, augment=False)
    X_train, y_train = _flatten_loader(train_loader)
    X_val, y_val = _flatten_loader(val_loader)
    X_test, y_test = _flatten_loader(test_loader)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=c, max_iter=max_iter, n_jobs=-1, solver="lbfgs")
    clf.fit(X_train_s, y_train)

    val_acc = accuracy_score(y_val, clf.predict(X_val_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)

    return {"val_acc": float(val_acc), "test_acc": float(test_acc), "save_path": save_path}
