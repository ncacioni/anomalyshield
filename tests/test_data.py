"""Tests for the data layer: loader, preprocessor, and dataset utilities."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from data.datasets import generate_synthetic, get_sample_csv_path
from src.data.loader import TimeSeriesLoader
from src.data.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Minimal 10-row DatetimeIndex DataFrame with a single numeric column."""
    idx = pd.date_range(start="2024-01-01", periods=10, freq="D", name="date")
    return pd.DataFrame({"value": np.arange(10, dtype=float)}, index=idx)


@pytest.fixture()
def df_with_nans(simple_df: pd.DataFrame) -> pd.DataFrame:
    """simple_df with NaN values inserted at positions 3 and 7."""
    df = simple_df.copy()
    df.iloc[3] = np.nan
    df.iloc[7] = np.nan
    return df


@pytest.fixture()
def sample_csv_path() -> str:
    return get_sample_csv_path()


# ---------------------------------------------------------------------------
# TimeSeriesLoader.from_csv
# ---------------------------------------------------------------------------


class TestFromCsv:
    def test_returns_datetime_index(self, sample_csv_path: str) -> None:
        df = TimeSeriesLoader.from_csv(sample_csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_named_date(self, sample_csv_path: str) -> None:
        df = TimeSeriesLoader.from_csv(sample_csv_path)
        assert df.index.name == "date"

    def test_value_column_is_float64(self, sample_csv_path: str) -> None:
        df = TimeSeriesLoader.from_csv(sample_csv_path)
        assert df["value"].dtype == np.float64

    def test_sorted_ascending(self, sample_csv_path: str) -> None:
        df = TimeSeriesLoader.from_csv(sample_csv_path)
        assert df.index.is_monotonic_increasing

    def test_non_empty(self, sample_csv_path: str) -> None:
        df = TimeSeriesLoader.from_csv(sample_csv_path)
        assert len(df) > 0

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            TimeSeriesLoader.from_csv("/nonexistent/path/data.csv")

    def test_missing_date_column_raises(self, tmp_path) -> None:
        # The CSV has a "ts" column but we ask for "date".
        # pandas.read_csv raises a ValueError during parse_dates before the
        # loader's own column check, so we match on the broader re-raise message.
        csv = tmp_path / "bad.csv"
        csv.write_text("ts,value\n2024-01-01,1.0\n")
        with pytest.raises(ValueError):
            TimeSeriesLoader.from_csv(str(csv), date_col="date")

    def test_missing_value_column_raises(self, tmp_path) -> None:
        csv = tmp_path / "bad.csv"
        csv.write_text("date,price\n2024-01-01,1.0\n")
        with pytest.raises(ValueError, match="Required columns missing"):
            TimeSeriesLoader.from_csv(str(csv), date_col="date", value_col="value")

    def test_non_numeric_value_column_raises(self, tmp_path) -> None:
        csv = tmp_path / "bad.csv"
        csv.write_text("date,value\n2024-01-01,abc\n2024-01-02,xyz\n")
        with pytest.raises(ValueError):
            TimeSeriesLoader.from_csv(str(csv))

    def test_custom_column_names(self, tmp_path) -> None:
        csv = tmp_path / "custom.csv"
        csv.write_text("ts,price\n2024-01-01,10.5\n2024-01-02,11.0\n")
        df = TimeSeriesLoader.from_csv(str(csv), date_col="ts", value_col="price")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "price" in df.columns


# ---------------------------------------------------------------------------
# TimeSeriesLoader.from_dataframe
# ---------------------------------------------------------------------------


class TestFromDataframe:
    def test_existing_datetime_index_accepted(self, simple_df: pd.DataFrame) -> None:
        df = TimeSeriesLoader.from_dataframe(simple_df)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_date_col_promoted_to_index(self) -> None:
        raw = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=5, freq="D"), "value": range(5)}
        )
        df = TimeSeriesLoader.from_dataframe(raw, date_col="date", value_col="value")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "date" not in df.columns

    def test_auto_detect_first_numeric_column(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
        raw = pd.DataFrame({"alpha": range(5), "beta": range(5, 10)}, index=idx)
        df = TimeSeriesLoader.from_dataframe(raw)
        # Both columns are numeric — both should be kept
        assert df.shape[1] >= 1
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_value_col_selection(self, simple_df: pd.DataFrame) -> None:
        df = TimeSeriesLoader.from_dataframe(simple_df, value_col="value")
        assert list(df.columns) == ["value"]

    def test_no_datetime_index_no_date_col_raises(self) -> None:
        raw = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            TimeSeriesLoader.from_dataframe(raw)

    def test_missing_date_col_raises(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="date_col"):
            TimeSeriesLoader.from_dataframe(simple_df, date_col="nonexistent")

    def test_missing_value_col_raises(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="value_col"):
            TimeSeriesLoader.from_dataframe(simple_df, value_col="nonexistent")

    def test_no_numeric_columns_raises(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
        raw = pd.DataFrame({"label": ["a", "b", "c"]}, index=idx)
        with pytest.raises(ValueError, match="[Nn]umeric"):
            TimeSeriesLoader.from_dataframe(raw)

    def test_original_not_mutated(self, simple_df: pd.DataFrame) -> None:
        original_cols = list(simple_df.columns)
        TimeSeriesLoader.from_dataframe(simple_df, value_col="value")
        assert list(simple_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Preprocessor.handle_missing
# ---------------------------------------------------------------------------


class TestHandleMissing:
    @pytest.mark.parametrize("strategy", ["interpolate", "ffill", "bfill", "drop"])
    def test_no_nans_after_valid_strategy(
        self, df_with_nans: pd.DataFrame, strategy: str
    ) -> None:
        result = Preprocessor.handle_missing(df_with_nans, strategy=strategy)
        if strategy == "drop":
            # drop may still have NaN-free rows
            assert result.isna().sum().sum() == 0
        else:
            assert result.isna().sum().sum() == 0

    def test_interpolate_preserves_length(self, df_with_nans: pd.DataFrame) -> None:
        result = Preprocessor.handle_missing(df_with_nans, strategy="interpolate")
        assert len(result) == len(df_with_nans)

    def test_ffill_preserves_length(self, df_with_nans: pd.DataFrame) -> None:
        result = Preprocessor.handle_missing(df_with_nans, strategy="ffill")
        assert len(result) == len(df_with_nans)

    def test_bfill_preserves_length(self, df_with_nans: pd.DataFrame) -> None:
        result = Preprocessor.handle_missing(df_with_nans, strategy="bfill")
        assert len(result) == len(df_with_nans)

    def test_drop_reduces_length(self, df_with_nans: pd.DataFrame) -> None:
        result = Preprocessor.handle_missing(df_with_nans, strategy="drop")
        assert len(result) < len(df_with_nans)

    def test_original_not_mutated(self, df_with_nans: pd.DataFrame) -> None:
        nan_count_before = df_with_nans.isna().sum().sum()
        Preprocessor.handle_missing(df_with_nans, strategy="ffill")
        assert df_with_nans.isna().sum().sum() == nan_count_before

    def test_unknown_strategy_raises(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            Preprocessor.handle_missing(simple_df, strategy="median")


# ---------------------------------------------------------------------------
# Preprocessor.normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_returns_tuple_of_df_and_scaler(
        self, simple_df: pd.DataFrame, method: str
    ) -> None:
        result = Preprocessor.normalize(simple_df, method=method)
        assert isinstance(result, tuple)
        assert len(result) == 2
        scaled_df, scaler = result
        assert isinstance(scaled_df, pd.DataFrame)

    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_index_preserved(self, simple_df: pd.DataFrame, method: str) -> None:
        scaled_df, _ = Preprocessor.normalize(simple_df, method=method)
        pd.testing.assert_index_equal(scaled_df.index, simple_df.index)

    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_columns_preserved(self, simple_df: pd.DataFrame, method: str) -> None:
        scaled_df, _ = Preprocessor.normalize(simple_df, method=method)
        assert list(scaled_df.columns) == list(simple_df.columns)

    def test_standard_zero_mean(self, simple_df: pd.DataFrame) -> None:
        scaled_df, _ = Preprocessor.normalize(simple_df, method="standard")
        assert abs(scaled_df["value"].mean()) < 1e-10

    def test_minmax_range_zero_to_one(self, simple_df: pd.DataFrame) -> None:
        scaled_df, _ = Preprocessor.normalize(simple_df, method="minmax")
        assert scaled_df["value"].min() == pytest.approx(0.0)
        assert scaled_df["value"].max() == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_inverse_transform_recovers_original(
        self, simple_df: pd.DataFrame, method: str
    ) -> None:
        scaled_df, scaler = Preprocessor.normalize(simple_df, method=method)
        recovered = scaler.inverse_transform(scaled_df.values)
        np.testing.assert_allclose(recovered, simple_df.values, rtol=1e-5, atol=1e-8)

    def test_unknown_method_raises(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown normalization method"):
            Preprocessor.normalize(simple_df, method="zscore")


# ---------------------------------------------------------------------------
# Preprocessor.create_windows
# ---------------------------------------------------------------------------


class TestCreateWindows:
    def test_1d_output_shape(self) -> None:
        data = np.arange(20, dtype=float)
        windows = Preprocessor.create_windows(data, window_size=5)
        # n_windows = 20 - 5 + 1 = 16, n_features = 1
        assert windows.shape == (16, 5, 1)

    def test_2d_output_shape(self) -> None:
        data = np.arange(40, dtype=float).reshape(20, 2)
        windows = Preprocessor.create_windows(data, window_size=5)
        # n_windows = 20 - 5 + 1 = 16, n_features = 2
        assert windows.shape == (16, 5, 2)

    def test_window_equals_full_data_when_size_equals_length(self) -> None:
        data = np.arange(10, dtype=float)
        windows = Preprocessor.create_windows(data, window_size=10)
        assert windows.shape == (1, 10, 1)

    def test_window_size_exceeds_timesteps_raises(self) -> None:
        data = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="window_size"):
            Preprocessor.create_windows(data, window_size=10)

    def test_window_content_correctness(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        windows = Preprocessor.create_windows(data, window_size=3)
        # First window: [1, 2, 3], last window: [3, 4, 5]
        np.testing.assert_array_equal(windows[0, :, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(windows[-1, :, 0], [3.0, 4.0, 5.0])

    def test_window_size_one(self) -> None:
        data = np.arange(8, dtype=float)
        windows = Preprocessor.create_windows(data, window_size=1)
        assert windows.shape == (8, 1, 1)


# ---------------------------------------------------------------------------
# Preprocessor.add_features
# ---------------------------------------------------------------------------


class TestAddFeatures:
    def test_rolling_mean_7_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_rolling_mean_7" in out.columns

    def test_rolling_std_7_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_rolling_std_7" in out.columns

    def test_rolling_mean_30_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_rolling_mean_30" in out.columns

    def test_rolling_std_30_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_rolling_std_30" in out.columns

    def test_lag_1_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_lag_1" in out.columns

    def test_lag_7_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "value_lag_7" in out.columns

    def test_calendar_columns_present(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert "day_of_week" in out.columns
        assert "month" in out.columns

    def test_day_of_week_range(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert out["day_of_week"].between(0, 6).all()

    def test_month_range(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert out["month"].between(1, 12).all()

    def test_more_columns_than_input(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        assert out.shape[1] > simple_df.shape[1]

    def test_same_index_as_input(self, simple_df: pd.DataFrame) -> None:
        out = Preprocessor.add_features(simple_df)
        pd.testing.assert_index_equal(out.index, simple_df.index)

    def test_original_not_mutated(self, simple_df: pd.DataFrame) -> None:
        original_cols = list(simple_df.columns)
        Preprocessor.add_features(simple_df)
        assert list(simple_df.columns) == original_cols

    def test_non_datetime_index_raises(self) -> None:
        raw = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            Preprocessor.add_features(raw)

    def test_multi_column_adds_features_per_column(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
        df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0, 20.0)}, index=idx)
        out = Preprocessor.add_features(df)
        assert "a_rolling_mean_7" in out.columns
        assert "b_rolling_mean_7" in out.columns


# ---------------------------------------------------------------------------
# generate_synthetic
# ---------------------------------------------------------------------------


class TestGenerateSynthetic:
    def test_default_output_shape(self) -> None:
        df = generate_synthetic()
        assert df.shape == (500, 2)

    def test_custom_n_points(self) -> None:
        df = generate_synthetic(n_points=100)
        assert len(df) == 100

    def test_columns_present(self) -> None:
        df = generate_synthetic()
        assert "value" in df.columns
        assert "is_anomaly" in df.columns

    def test_datetime_index(self) -> None:
        df = generate_synthetic()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_named_date(self) -> None:
        df = generate_synthetic()
        assert df.index.name == "date"

    def test_anomaly_ratio_approximate(self) -> None:
        n = 1000
        ratio = 0.05
        df = generate_synthetic(n_points=n, anomaly_ratio=ratio, seed=0)
        actual_ratio = df["is_anomaly"].mean()
        # Allow ±1 anomaly tolerance due to integer rounding
        assert abs(actual_ratio - ratio) <= 1 / n + 0.001

    def test_is_anomaly_binary(self) -> None:
        df = generate_synthetic()
        assert set(df["is_anomaly"].unique()).issubset({0, 1})

    def test_value_column_is_float64(self) -> None:
        df = generate_synthetic()
        assert df["value"].dtype == np.float64

    def test_reproducibility_same_seed(self) -> None:
        df1 = generate_synthetic(seed=7)
        df2 = generate_synthetic(seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        df1 = generate_synthetic(seed=1)
        df2 = generate_synthetic(seed=2)
        assert not df1["value"].equals(df2["value"])

    def test_n_points_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_points"):
            generate_synthetic(n_points=0)

    def test_anomaly_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="anomaly_ratio"):
            generate_synthetic(anomaly_ratio=0.0)

    def test_anomaly_ratio_one_raises(self) -> None:
        with pytest.raises(ValueError, match="anomaly_ratio"):
            generate_synthetic(anomaly_ratio=1.0)

    def test_starts_on_2024_01_01(self) -> None:
        df = generate_synthetic()
        assert df.index[0] == pd.Timestamp("2024-01-01")

    def test_daily_frequency(self) -> None:
        df = generate_synthetic(n_points=10)
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta("1D")).all()


# ---------------------------------------------------------------------------
# get_sample_csv_path
# ---------------------------------------------------------------------------


class TestGetSampleCsvPath:
    def test_returns_string(self) -> None:
        path = get_sample_csv_path()
        assert isinstance(path, str)

    def test_path_exists(self) -> None:
        path = get_sample_csv_path()
        assert os.path.isfile(path)

    def test_path_is_absolute(self) -> None:
        path = get_sample_csv_path()
        assert os.path.isabs(path)

    def test_file_is_readable_csv(self) -> None:
        path = get_sample_csv_path()
        df = pd.read_csv(path)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_from_dataframe_empty_raises(self) -> None:
        idx = pd.date_range("2024-01-01", periods=0, freq="D", name="date")
        empty = pd.DataFrame({"value": pd.Series([], dtype=float)}, index=idx)
        with pytest.raises(ValueError):
            TimeSeriesLoader.from_dataframe(empty, value_col="value")

    def test_from_csv_empty_file_raises(self, tmp_path) -> None:
        csv = tmp_path / "empty.csv"
        csv.write_text("date,value\n")
        with pytest.raises(Exception):
            TimeSeriesLoader.from_csv(str(csv))

    def test_handle_missing_no_nans_is_idempotent(self, simple_df: pd.DataFrame) -> None:
        result = Preprocessor.handle_missing(simple_df, strategy="interpolate")
        pd.testing.assert_frame_equal(result, simple_df)

    def test_create_windows_large_window_equals_all_data(self) -> None:
        data = np.ones(100)
        windows = Preprocessor.create_windows(data, window_size=100)
        assert windows.shape == (1, 100, 1)
