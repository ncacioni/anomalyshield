"""Time series preprocessing — missing value handling, normalization, windowing, feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class Preprocessor:
    """Stateless helper that transforms time series DataFrames.

    Each method accepts a DataFrame and returns a transformed copy — the
    original is never modified in-place.
    """

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    @staticmethod
    def handle_missing(
        df: pd.DataFrame,
        strategy: str = "interpolate",
    ) -> pd.DataFrame:
        """Fill or remove missing values.

        Parameters
        ----------
        df:
            Input DataFrame (DatetimeIndex expected).
        strategy:
            One of:
            - ``"interpolate"`` — linear interpolation along the time axis.
            - ``"ffill"``       — forward-fill (last observed value).
            - ``"bfill"``       — backward-fill (next observed value).
            - ``"drop"``        — drop rows that contain any NaN.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.

        Raises
        ------
        ValueError
            If an unsupported strategy is supplied.
        """
        valid = {"interpolate", "ffill", "bfill", "drop"}
        if strategy not in valid:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Valid options: {sorted(valid)}"
            )

        df = df.copy()

        if strategy == "interpolate":
            df = df.interpolate(method="time")
            # Edge NaNs at boundaries are not covered by time interpolation
            df = df.ffill().bfill()
        elif strategy == "ffill":
            df = df.ffill()
        elif strategy == "bfill":
            df = df.bfill()
        elif strategy == "drop":
            df = df.dropna()

        return df

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        method: str = "standard",
    ) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | RobustScaler]:
        """Scale numeric columns to a common range.

        Parameters
        ----------
        df:
            Input DataFrame with numeric columns.
        method:
            One of:
            - ``"standard"`` — zero mean, unit variance (StandardScaler).
            - ``"minmax"``   — scales to [0, 1] (MinMaxScaler).
            - ``"robust"``   — median-centered, IQR-scaled (RobustScaler).

        Returns
        -------
        tuple[pd.DataFrame, scaler]
            A pair of ``(scaled_df, fitted_scaler)``.  Call
            ``scaler.inverse_transform(scaled_df.values)`` to recover
            the original scale.

        Raises
        ------
        ValueError
            If an unsupported method is supplied.
        """
        valid = {"standard", "minmax", "robust"}
        if method not in valid:
            raise ValueError(
                f"Unknown normalization method '{method}'. Valid options: {sorted(valid)}"
            )

        scaler_cls = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }[method]

        scaler = scaler_cls()
        scaled_values = scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

        return scaled_df, scaler

    # ------------------------------------------------------------------
    # Sliding windows
    # ------------------------------------------------------------------

    @staticmethod
    def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
        """Create sliding windows suitable for sequence models.

        Parameters
        ----------
        data:
            1-D or 2-D array of shape ``(n_timesteps,)`` or
            ``(n_timesteps, n_features)``.
        window_size:
            Number of timesteps per window.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_windows, window_size, n_features)`` where
            ``n_windows = n_timesteps - window_size + 1``.

        Raises
        ------
        ValueError
            If *window_size* is larger than the number of timesteps.
        """
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_timesteps, n_features = arr.shape

        if window_size > n_timesteps:
            raise ValueError(
                f"window_size ({window_size}) exceeds number of timesteps ({n_timesteps})."
            )

        n_windows = n_timesteps - window_size + 1
        windows = np.lib.stride_tricks.sliding_window_view(arr, (window_size, n_features))
        # sliding_window_view returns shape (n_windows, 1, window_size, n_features)
        windows = windows.reshape(n_windows, window_size, n_features)

        return windows

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Append rolling, lag, and calendar features.

        New columns added for each numeric column *col*:
        - ``{col}_rolling_mean_7``, ``{col}_rolling_std_7``
        - ``{col}_rolling_mean_30``, ``{col}_rolling_std_30``
        - ``{col}_lag_1``, ``{col}_lag_7``

        Calendar features (computed once, not per column):
        - ``day_of_week`` — integer 0 (Monday) to 6 (Sunday)
        - ``month``        — integer 1–12

        Parameters
        ----------
        df:
            DataFrame with a DatetimeIndex and numeric columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the additional feature columns appended.

        Raises
        ------
        ValueError
            If *df* does not have a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex to add time-based features.")

        out = df.copy()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            series = df[col]
            out[f"{col}_rolling_mean_7"] = series.rolling(window=7, min_periods=1).mean()
            out[f"{col}_rolling_std_7"] = series.rolling(window=7, min_periods=1).std()
            out[f"{col}_rolling_mean_30"] = series.rolling(window=30, min_periods=1).mean()
            out[f"{col}_rolling_std_30"] = series.rolling(window=30, min_periods=1).std()
            out[f"{col}_lag_1"] = series.shift(1)
            out[f"{col}_lag_7"] = series.shift(7)

        out["day_of_week"] = df.index.day_of_week.astype("int8")
        out["month"] = df.index.month.astype("int8")

        return out
