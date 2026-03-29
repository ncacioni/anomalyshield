"""Time series data loader — CSV and DataFrame ingestion with validation."""

from __future__ import annotations

import os

import pandas as pd

# Maximum CSV file size accepted by from_csv (50 MB).
MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024


class TimeSeriesLoader:
    """Loads and standardizes time series data from various sources.

    All returned DataFrames have a DatetimeIndex (UTC-aware or naive, as
    supplied) sorted in ascending order, with numeric value columns.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def from_csv(
        path: str,
        date_col: str = "date",
        value_col: str = "value",
    ) -> pd.DataFrame:
        """Read a CSV file and return a validated time series DataFrame.

        Parameters
        ----------
        path:
            Filesystem path to the CSV file.
        date_col:
            Name of the column that contains date/datetime strings.
        value_col:
            Name of the column that contains numeric values.

        Returns
        -------
        pd.DataFrame
            DataFrame with a DatetimeIndex and the value column cast to float64.

        Raises
        ------
        FileNotFoundError
            If *path* does not point to an existing file.
        ValueError
            If required columns are missing or values cannot be parsed.
        """
        # Validate file exists and is a CSV
        resolved = os.path.realpath(path)
        if not resolved.lower().endswith(".csv"):
            raise ValueError("Only .csv files are accepted.")
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"CSV file not found: {path}")
        if os.path.getsize(resolved) > MAX_CSV_SIZE_BYTES:
            raise ValueError(
                f"CSV file exceeds maximum size of "
                f"{MAX_CSV_SIZE_BYTES // (1024 * 1024)}MB."
            )

        try:
            df = pd.read_csv(resolved, parse_dates=[date_col])
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {path}")
        except ValueError as exc:
            raise ValueError(f"Failed to parse CSV at '{path}': {exc}") from exc

        return TimeSeriesLoader._validate_and_standardize(df, date_col, value_col)

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        date_col: str | None = None,
        value_col: str | None = None,
    ) -> pd.DataFrame:
        """Standardize an existing DataFrame into a validated time series.

        If the DataFrame already carries a DatetimeIndex and *date_col* is
        ``None``, the index is used as-is.  If *value_col* is ``None`` the
        first numeric column is used.

        Parameters
        ----------
        df:
            Source DataFrame.
        date_col:
            Column name to promote to DatetimeIndex.  Pass ``None`` when the
            DataFrame already has a DatetimeIndex.
        value_col:
            Column name that holds numeric values.  Pass ``None`` to auto-
            detect the first numeric column.

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with DatetimeIndex.

        Raises
        ------
        ValueError
            If no datetime index can be established or no numeric column found.
        """
        df = df.copy()

        # Resolve date column
        if date_col is not None:
            if date_col not in df.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame columns.")
            df = df.set_index(pd.to_datetime(df[date_col]))
            df.index.name = "date"
            df = df.drop(columns=[date_col])
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame does not have a DatetimeIndex and no 'date_col' was provided."
            )

        # Resolve value column
        if value_col is not None:
            if value_col not in df.columns:
                raise ValueError(f"value_col '{value_col}' not found in DataFrame columns.")
            df = df[[value_col]]
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in DataFrame.")
            df = df[numeric_cols]

        return TimeSeriesLoader._finalize(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_standardize(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Shared validation path for CSV-sourced data."""
        missing = [c for c in (date_col, value_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing from CSV: {missing}")

        # Promote date column to DatetimeIndex
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        df = df.set_index(df[date_col])
        df.index.name = "date"
        df = df.drop(columns=[date_col])

        # Cast value column to float64
        try:
            df[value_col] = pd.to_numeric(df[value_col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{value_col}' contains non-numeric values: {exc}"
            ) from exc

        # Keep only the value column (plus any remaining columns from CSV)
        cols = [value_col] + [c for c in df.columns if c != value_col]
        df = df[cols]

        return TimeSeriesLoader._finalize(df)

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        """Sort by index, validate DatetimeIndex, ensure numeric values."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index is not a DatetimeIndex after processing.")

        df = df.sort_index()

        # Validate that all remaining columns are numeric
        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns remain after processing: {non_numeric}. "
                "Pass explicit value_col to select only the numeric series."
            )

        if df.empty:
            raise ValueError("Resulting DataFrame is empty.")

        return df
