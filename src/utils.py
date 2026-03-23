"""Utility functions for AnomalyShield evaluation, comparison, and reproducibility."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels. Accepts -1/1 (anomaly/normal) or 0/1 (anomaly/normal).
        Internally converted to -1/1 for consistency.
    y_pred : np.ndarray
        Predicted labels in -1/1 format.
    y_scores : np.ndarray | None
        Anomaly scores (higher = more anomalous). If provided, AUC-ROC is computed.

    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision, recall, f1, and optionally auc_roc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert 0/1 labels to -1/1 if needed (0 -> anomaly -> -1, 1 -> normal -> 1)
    if set(np.unique(y_true)).issubset({0, 1}):
        y_true = np.where(y_true == 0, -1, 1)

    # For sklearn metrics, treat -1 (anomaly) as the positive class
    # Convert to binary: anomaly=1, normal=0 for metric computation
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }

    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true_bin, y_scores))
        except ValueError:
            # Single class in y_true — AUC undefined
            pass

    return metrics


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from AnomalyShield results.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of detector name to result dict. Each result dict must contain
        a ``metrics`` key with a metrics dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with detector names as the index and metric names as columns.

    Raises
    ------
    ValueError
        If no results contain metrics (y_true was not provided during run_all).
    """
    rows: dict[str, dict] = {}
    for name, result in results.items():
        if "metrics" in result and result["metrics"] is not None:
            rows[name] = result["metrics"]

    if not rows:
        raise ValueError(
            "No metrics available. Provide y_true when calling run_all to compute metrics."
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "detector"
    return df


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across common libraries.

    Sets seeds for:
    - Python's built-in ``random`` module
    - NumPy's random generator
    - PyTorch (if available)

    Parameters
    ----------
    seed : int
        Seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
