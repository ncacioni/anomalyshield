"""Utility functions for AnomalyShield evaluation, comparison, and reproducibility."""

from __future__ import annotations

import random
from datetime import datetime

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


def generate_report(results: dict, output_path: str | None = None) -> str:
    """Generate a Markdown report from AnomalyShield results.

    Parameters
    ----------
    results : dict
        Mapping of detector name to result dict as returned by
        ``AnomalyShield.run_all()``. Each value contains ``predictions``
        (np.ndarray of -1/1), ``scores`` (np.ndarray), and optionally
        ``metrics`` (dict).
    output_path : str | None
        If provided, the report is written to this file path.

    Returns
    -------
    str
        The full Markdown report.
    """
    lines: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Header ---
    lines.append("# AnomalyShield Detection Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    # --- Summary ---
    n_detectors = len(results)
    # Determine total data points from the first detector's predictions
    first_result = next(iter(results.values()))
    n_points = len(first_result["predictions"])

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Detectors run:** {n_detectors}")
    lines.append(f"- **Total data points:** {n_points}")
    lines.append("")

    anomaly_counts: dict[str, int] = {}
    for name, result in results.items():
        preds = np.asarray(result["predictions"])
        count = int(np.sum(preds == -1))
        anomaly_counts[name] = count
        lines.append(f"- **{name}:** {count} anomalies detected")

    lines.append("")

    # --- Metrics Table ---
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    has_metrics = any(
        "metrics" in r and r["metrics"] is not None for r in results.values()
    )

    if has_metrics:
        lines.append("## Metrics Comparison")
        lines.append("")

        # Build header row
        header = "| Detector | " + " | ".join(metric_keys) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(metric_keys)) + " |"
        lines.append(header)
        lines.append(separator)

        for name, result in results.items():
            metrics = result.get("metrics")
            if metrics is None:
                continue
            row_values = []
            for key in metric_keys:
                value = metrics.get(key)
                if value is not None:
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append("N/A")
            lines.append(f"| {name} | " + " | ".join(row_values) + " |")

        lines.append("")

    # --- Per-Detector Details ---
    lines.append("## Per-Detector Details")
    lines.append("")

    for name, result in results.items():
        preds = np.asarray(result["predictions"])
        count = anomaly_counts[name]
        pct = (count / n_points) * 100 if n_points > 0 else 0.0

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Anomalies detected:** {count}")
        lines.append(f"- **Percentage flagged:** {pct:.2f}%")

        metrics = result.get("metrics")
        if metrics is not None:
            for key in metric_keys:
                value = metrics.get(key)
                if value is not None:
                    lines.append(f"- **{key}:** {value:.4f}")

        lines.append("")

    # --- Ensemble Summary ---
    lines.append("## Ensemble Summary")
    lines.append("")

    all_preds = np.array([result["predictions"] for result in results.values()])
    anomaly_votes = np.sum(all_preds == -1, axis=0)

    majority_count = int(np.sum(anomaly_votes > n_detectors / 2))
    unanimous_count = int(np.sum(anomaly_votes == n_detectors))

    lines.append(
        f"- **Majority vote anomalies** (>{n_detectors // 2} detectors agree): "
        f"{majority_count}"
    )
    lines.append(
        f"- **Unanimous anomalies** (all {n_detectors} detectors agree): "
        f"{unanimous_count}"
    )
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
