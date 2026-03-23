"""Synthetic dataset generation and sample data path resolution."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def generate_synthetic(
    n_points: int = 500,
    anomaly_ratio: float = 0.02,
    noise_level: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic time series with trend, seasonality, noise, and anomalies.

    The series starts on 2024-01-01 with daily frequency.

    Parameters
    ----------
    n_points:
        Total number of data points.
    anomaly_ratio:
        Fraction of points to mark as anomalies (0 < anomaly_ratio < 1).
    noise_level:
        Standard deviation of the Gaussian noise component.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - DatetimeIndex named ``date``
        - ``value``      — float64 time series value
        - ``is_anomaly`` — int8 indicator (1 = anomaly, 0 = normal)

    Raises
    ------
    ValueError
        If *n_points* < 1 or *anomaly_ratio* is outside (0, 1).
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")
    if not (0 < anomaly_ratio < 1):
        raise ValueError(
            f"anomaly_ratio must be in (0, 1), got {anomaly_ratio}."
        )

    rng = np.random.default_rng(seed)

    t = np.arange(n_points, dtype=float)

    # Linear trend
    trend = 0.05 * t

    # Weekly + annual seasonality
    weekly = 5.0 * np.sin(2 * np.pi * t / 7)
    annual = 10.0 * np.sin(2 * np.pi * t / 365)

    # Gaussian noise
    noise = rng.normal(loc=0.0, scale=noise_level, size=n_points)

    values = trend + weekly + annual + noise + 100.0

    # Inject anomalies
    n_anomalies = max(1, int(n_points * anomaly_ratio))
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)
    is_anomaly = np.zeros(n_points, dtype=np.int8)

    for idx in anomaly_indices:
        # Spike: 4–8 standard deviations above or below the local value
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(4.0, 8.0) * noise_level
        values[idx] += direction * magnitude
        is_anomaly[idx] = 1

    date_index = pd.date_range(start="2024-01-01", periods=n_points, freq="D", name="date")

    df = pd.DataFrame(
        {
            "value": values.astype(np.float64),
            "is_anomaly": is_anomaly,
        },
        index=date_index,
    )

    return df


def get_sample_csv_path() -> str:
    """Return the absolute path to the bundled sample CSV file.

    The file is located at ``assets/sample_data.csv`` relative to the
    project root (the directory that contains this ``data/`` package).

    Returns
    -------
    str
        Absolute filesystem path to ``assets/sample_data.csv``.

    Raises
    ------
    FileNotFoundError
        If the sample CSV file does not exist at the expected location.
    """
    # This file lives at <project_root>/data/datasets.py
    # so the project root is one level up.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "assets", "sample_data.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Sample CSV not found at expected path: {csv_path}"
        )

    return csv_path
