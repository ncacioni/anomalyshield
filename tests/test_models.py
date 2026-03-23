"""Tests for AnomalyShield models: LOF, EllipticEnvelope, Autoencoder, Prophet, and generate_report."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normal_data() -> np.ndarray:
    """200 samples from a tight 2D Gaussian cluster centred at origin."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(200, 2))


@pytest.fixture(scope="module")
def anomaly_data() -> np.ndarray:
    """20 samples far from origin (clearly anomalous)."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=10.0, scale=0.5, size=(20, 2))


@pytest.fixture(scope="module")
def mixed_data(normal_data: np.ndarray, anomaly_data: np.ndarray) -> np.ndarray:
    """220 samples: 200 normal + 20 anomalous, stacked."""
    return np.vstack([normal_data, anomaly_data])


@pytest.fixture(scope="module")
def y_true_mixed(normal_data: np.ndarray, anomaly_data: np.ndarray) -> np.ndarray:
    """Ground truth labels: 1 for normal, -1 for anomaly."""
    normal_labels = np.ones(len(normal_data), dtype=int)
    anomaly_labels = -np.ones(len(anomaly_data), dtype=int)
    return np.concatenate([normal_labels, anomaly_labels])


@pytest.fixture(scope="module")
def time_series_df() -> pd.DataFrame:
    """100-point daily time series with a 'value' column and DatetimeIndex."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = rng.normal(loc=50.0, scale=5.0, size=100)
    df = pd.DataFrame({"value": values}, index=dates)
    return df


# ---------------------------------------------------------------------------
# LOFDetector
# ---------------------------------------------------------------------------


class TestLOFDetectorFit:
    def test_returns_self(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        result = det.fit(mixed_data)
        assert result is det

    def test_is_fitted_true_after_fit(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        det.fit(mixed_data)
        assert det.is_fitted is True

    def test_not_fitted_on_init(self) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector()
        assert det.is_fitted is False


def _lof_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (mixed, labels) suitable for LOF testing.

    LOF detects local density deviations. Clustered anomalies defeat LOF because
    intra-cluster density is high. We use scattered isolated anomaly points so
    each anomaly's nearest neighbors are drawn from the normal cluster, ensuring
    LOF >> 1 for each anomaly.
    """
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=0.5, size=(200, 2))
    # Scattered anomalies — each point isolated far from every other point
    rng2 = np.random.default_rng(7)
    angles = rng2.uniform(0, 2 * np.pi, size=20)
    radii = rng2.uniform(8.0, 12.0, size=20)
    anomaly_x = radii * np.cos(angles)
    anomaly_y = radii * np.sin(angles)
    anomaly = np.column_stack([anomaly_x, anomaly_y])
    mixed = np.vstack([normal, anomaly])
    return mixed, anomaly


class TestLOFDetectorPredict:
    def test_output_shape(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        det.fit(mixed_data)
        preds = det.predict(mixed_data)
        assert preds.shape == (len(mixed_data),)

    def test_labels_are_minus1_or_1(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        det.fit(mixed_data)
        preds = det.predict(mixed_data)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_predict_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict(mixed_data)

    def test_anomaly_points_mostly_flagged(self) -> None:
        """Isolated anomaly points scattered far from the normal cluster are mostly -1."""
        from src.models.lof import LOFDetector

        mixed, anomaly = _lof_dataset()
        det = LOFDetector(contamination=0.1, n_neighbors=10)
        det.fit(mixed)
        preds = det.predict(anomaly)
        anomaly_rate = np.mean(preds == -1)
        assert anomaly_rate > 0.5

    def test_normal_cluster_mostly_normal(self, mixed_data: np.ndarray) -> None:
        """On the training set the first 200 (normal) samples are mostly 1."""
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        det.fit(mixed_data)
        preds = det.predict(mixed_data)
        normal_rate = np.mean(preds[:200] == 1)
        assert normal_rate > 0.85


class TestLOFDetectorScoreSamples:
    def test_output_shape(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        det.fit(mixed_data)
        scores = det.score_samples(mixed_data)
        assert scores.shape == (len(mixed_data),)

    def test_anomalies_score_higher_on_average(self) -> None:
        """Isolated anomaly points have higher LOF scores than the normal cluster."""
        from src.models.lof import LOFDetector

        mixed, anomaly = _lof_dataset()
        normal = mixed[:200]
        det = LOFDetector(contamination=0.1, n_neighbors=10)
        det.fit(mixed)
        normal_scores = det.score_samples(normal)
        anomaly_scores = det.score_samples(anomaly)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_score_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.score_samples(mixed_data)


class TestLOFDetectorParams:
    def test_name_stored(self) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(name="my_lof")
        assert det.name == "my_lof"

    def test_default_name(self) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector()
        assert det.name == "LOF"

    def test_params_stored(self) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(n_neighbors=15, contamination=0.05)
        assert det.params["n_neighbors"] == 15
        assert det.params["contamination"] == 0.05

    def test_fit_predict_returns_correct_labels(self, mixed_data: np.ndarray) -> None:
        from src.models.lof import LOFDetector

        det = LOFDetector(contamination=0.1)
        preds = det.fit_predict(mixed_data)
        assert preds.shape == (len(mixed_data),)
        assert set(np.unique(preds)).issubset({-1, 1})


# ---------------------------------------------------------------------------
# EllipticEnvelopeDetector
# ---------------------------------------------------------------------------


class TestEllipticEnvelopeDetectorFit:
    def test_returns_self(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        result = det.fit(mixed_data)
        assert result is det

    def test_is_fitted_true_after_fit(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        assert det.is_fitted is True

    def test_not_fitted_on_init(self) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector()
        assert det.is_fitted is False


class TestEllipticEnvelopeDetectorPredict:
    def test_output_shape(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        preds = det.predict(mixed_data)
        assert preds.shape == (len(mixed_data),)

    def test_labels_are_minus1_or_1(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        preds = det.predict(mixed_data)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_predict_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict(mixed_data)

    def test_anomaly_cluster_mostly_flagged(
        self, mixed_data: np.ndarray, anomaly_data: np.ndarray
    ) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        preds = det.predict(anomaly_data)
        anomaly_rate = np.mean(preds == -1)
        assert anomaly_rate > 0.5

    def test_normal_cluster_mostly_normal(
        self, mixed_data: np.ndarray, normal_data: np.ndarray
    ) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        preds = det.predict(normal_data)
        normal_rate = np.mean(preds == 1)
        assert normal_rate > 0.85


class TestEllipticEnvelopeDetectorScoreSamples:
    def test_output_shape(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        scores = det.score_samples(mixed_data)
        assert scores.shape == (len(mixed_data),)

    def test_anomalies_score_higher_on_average(
        self, mixed_data: np.ndarray, normal_data: np.ndarray, anomaly_data: np.ndarray
    ) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det.fit(mixed_data)
        normal_scores = det.score_samples(normal_data)
        anomaly_scores = det.score_samples(anomaly_data)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_score_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.score_samples(mixed_data)


class TestEllipticEnvelopeDetectorReproducibility:
    def test_same_seed_same_predictions(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det1 = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        det2 = EllipticEnvelopeDetector(contamination=0.1, random_state=42)
        preds1 = det1.fit(mixed_data).predict(mixed_data)
        preds2 = det2.fit(mixed_data).predict(mixed_data)
        np.testing.assert_array_equal(preds1, preds2)

    def test_same_seed_same_scores(self, mixed_data: np.ndarray) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det1 = EllipticEnvelopeDetector(contamination=0.1, random_state=0)
        det2 = EllipticEnvelopeDetector(contamination=0.1, random_state=0)
        scores1 = det1.fit(mixed_data).score_samples(mixed_data)
        scores2 = det2.fit(mixed_data).score_samples(mixed_data)
        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_params_stored(self) -> None:
        from src.models.elliptic_envelope import EllipticEnvelopeDetector

        det = EllipticEnvelopeDetector(
            name="EE_test", contamination=0.05, random_state=7
        )
        assert det.name == "EE_test"
        assert det.params["contamination"] == 0.05
        assert det.params["random_state"] == 7


# ---------------------------------------------------------------------------
# AutoencoderDetector
# ---------------------------------------------------------------------------


# Small parameters shared across all autoencoder tests to keep runtime fast.
_AE_PARAMS = dict(hidden_dim=8, epochs=5, window_size=10, batch_size=16, random_state=42)


@pytest.fixture(scope="module")
def ae_normal_1d() -> np.ndarray:
    """150-point 1D normal time series."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(150,))


@pytest.fixture(scope="module")
def ae_normal_2d() -> np.ndarray:
    """150-point 2D normal time series."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(150, 2))


@pytest.fixture(scope="module")
def ae_anomaly_2d() -> np.ndarray:
    """20-point clearly anomalous 2D signal (large magnitude)."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=50.0, scale=1.0, size=(20, 2))


class TestAutoencoderDetectorFit:
    def test_returns_self(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        result = det.fit(ae_normal_2d)
        assert result is det

    def test_is_fitted_true_after_fit(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        assert det.is_fitted is True

    def test_not_fitted_on_init(self) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        assert det.is_fitted is False

    def test_threshold_set_after_fit(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        assert det._threshold is not None

    def test_accepts_1d_input(self, ae_normal_1d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_1d)
        assert det.is_fitted is True

    def test_too_few_samples_raises(self) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(window_size=20, **{k: v for k, v in _AE_PARAMS.items() if k != "window_size"})
        tiny = np.zeros((5, 2))
        with pytest.raises(ValueError, match="window_size"):
            det.fit(tiny)


class TestAutoencoderDetectorPredict:
    def test_output_shape_2d(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        preds = det.predict(ae_normal_2d)
        # windowed output has n_samples - window_size + 1 elements
        expected_len = len(ae_normal_2d) - _AE_PARAMS["window_size"] + 1
        assert preds.shape == (expected_len,)

    def test_output_shape_1d(self, ae_normal_1d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_1d)
        preds = det.predict(ae_normal_1d)
        expected_len = len(ae_normal_1d) - _AE_PARAMS["window_size"] + 1
        assert preds.shape == (expected_len,)

    def test_labels_are_minus1_or_1(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        preds = det.predict(ae_normal_2d)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_predict_before_fit_raises(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict(ae_normal_2d)


class TestAutoencoderDetectorScoreSamples:
    def test_output_shape(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        scores = det.score_samples(ae_normal_2d)
        expected_len = len(ae_normal_2d) - _AE_PARAMS["window_size"] + 1
        assert scores.shape == (expected_len,)

    def test_scores_are_non_negative(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        det.fit(ae_normal_2d)
        scores = det.score_samples(ae_normal_2d)
        # MSE reconstruction error is always >= 0
        assert np.all(scores >= 0)

    def test_score_before_fit_raises(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det = AutoencoderDetector(**_AE_PARAMS)
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.score_samples(ae_normal_2d)

    def test_anomalies_score_higher_than_normals(
        self, ae_normal_2d: np.ndarray, ae_anomaly_2d: np.ndarray
    ) -> None:
        """Anomalous windows should have higher reconstruction error on average."""
        from src.models.autoencoder import AutoencoderDetector

        # Use more epochs and larger data to make this reliable
        rng = np.random.default_rng(42)
        normal_train = rng.normal(loc=0.0, scale=0.5, size=(200, 2))
        anomaly_eval = rng.normal(loc=20.0, scale=0.5, size=(50, 2))

        det = AutoencoderDetector(
            hidden_dim=8, epochs=20, window_size=10, batch_size=16, random_state=42
        )
        det.fit(normal_train)

        normal_scores = det.score_samples(normal_train)
        anomaly_scores = det.score_samples(anomaly_eval)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)


class TestAutoencoderDetectorReproducibility:
    def test_same_seed_same_scores(self, ae_normal_2d: np.ndarray) -> None:
        from src.models.autoencoder import AutoencoderDetector

        det1 = AutoencoderDetector(**_AE_PARAMS)
        det2 = AutoencoderDetector(**_AE_PARAMS)
        scores1 = det1.fit(ae_normal_2d).score_samples(ae_normal_2d)
        scores2 = det2.fit(ae_normal_2d).score_samples(ae_normal_2d)
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)


# ---------------------------------------------------------------------------
# ProphetForecaster
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prophet_df() -> pd.DataFrame:
    """100-point daily time series with DatetimeIndex and 'value' column."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = rng.normal(loc=50.0, scale=5.0, size=100)
    df = pd.DataFrame({"value": values}, index=dates)
    return df


def _suppress_prophet_logging() -> None:
    """Suppress Prophet / cmdstanpy verbose output."""
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("stan").setLevel(logging.ERROR)


class TestProphetForecasterFit:
    def test_fit_returns_self(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.fit(prophet_df)
        assert result is forecaster

    def test_is_fitted_true_after_fit(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        forecaster.fit(prophet_df)
        assert forecaster.is_fitted is True

    def test_not_fitted_on_init(self) -> None:
        from src.models.prophet_model import ProphetForecaster

        forecaster = ProphetForecaster()
        assert forecaster.is_fitted is False

    def test_fit_requires_datetime_index(self) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster()
        df_bad = pd.DataFrame({"value": [1.0, 2.0, 3.0]})  # default integer index
        with pytest.raises(ValueError, match="DatetimeIndex"):
            forecaster.fit(df_bad)

    def test_fit_requires_value_column(self) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster()
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df_bad = pd.DataFrame({"price": [1.0] * 10}, index=dates)
        with pytest.raises(ValueError, match="'value' column"):
            forecaster.fit(df_bad)


class TestProphetForecasterPredict:
    @pytest.fixture(scope="class")
    def fitted_forecaster(self, prophet_df: pd.DataFrame):
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        forecaster.fit(prophet_df)
        return forecaster

    def test_predict_returns_dataframe(self, fitted_forecaster) -> None:
        forecast = fitted_forecaster.predict(periods=5)
        assert isinstance(forecast, pd.DataFrame)

    def test_predict_has_required_columns(self, fitted_forecaster) -> None:
        forecast = fitted_forecaster.predict(periods=5)
        for col in ("ds", "yhat", "yhat_lower", "yhat_upper"):
            assert col in forecast.columns

    def test_predict_row_count(self, fitted_forecaster, prophet_df: pd.DataFrame) -> None:
        periods = 5
        forecast = fitted_forecaster.predict(periods=periods)
        # Prophet returns history + future periods
        assert len(forecast) == len(prophet_df) + periods

    def test_predict_before_fit_raises(self) -> None:
        from src.models.prophet_model import ProphetForecaster

        forecaster = ProphetForecaster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            forecaster.predict(periods=5)

    def test_yhat_lower_le_yhat_le_yhat_upper(self, fitted_forecaster) -> None:
        forecast = fitted_forecaster.predict(periods=5)
        assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
        assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


class TestProphetForecasterDetectAnomalies:
    def test_detect_anomalies_returns_dataframe(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(prophet_df)
        assert isinstance(result, pd.DataFrame)

    def test_detect_anomalies_has_is_anomaly_column(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(prophet_df)
        assert "is_anomaly" in result.columns

    def test_detect_anomalies_has_forecast_columns(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(prophet_df)
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            assert col in result.columns

    def test_detect_anomalies_preserves_row_count(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(prophet_df)
        assert len(result) == len(prophet_df)

    def test_detect_anomalies_is_anomaly_is_bool(self, prophet_df: pd.DataFrame) -> None:
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(prophet_df)
        assert result["is_anomaly"].dtype == bool

    def test_detect_anomalies_flags_obvious_outliers(self) -> None:
        """Points far outside the uncertainty interval must be flagged."""
        from src.models.prophet_model import ProphetForecaster

        _suppress_prophet_logging()
        rng = np.random.default_rng(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        values = rng.normal(loc=50.0, scale=2.0, size=100)
        # Inject two extreme outliers
        values[50] = 500.0
        values[75] = -400.0
        df = pd.DataFrame({"value": values}, index=dates)

        forecaster = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False)
        result = forecaster.detect_anomalies(df)
        assert result["is_anomaly"].iloc[50] is np.bool_(True)
        assert result["is_anomaly"].iloc[75] is np.bool_(True)


# ---------------------------------------------------------------------------
# generate_report (utils.py)
# ---------------------------------------------------------------------------


def _make_results_no_metrics() -> dict:
    """Synthetic results dict without metrics (no y_true provided)."""
    rng = np.random.default_rng(0)
    preds = np.ones(100, dtype=int)
    preds[:10] = -1
    scores = rng.uniform(0, 1, 100)
    return {
        "DetectorA": {"predictions": preds, "scores": scores},
    }


def _make_results_with_metrics() -> dict:
    """Synthetic results dict with metrics."""
    rng = np.random.default_rng(0)
    preds = np.ones(100, dtype=int)
    preds[:10] = -1
    scores = rng.uniform(0, 1, 100)
    metrics = {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82,
        "auc_roc": 0.91,
    }
    return {
        "DetectorA": {"predictions": preds, "scores": scores, "metrics": metrics},
    }


class TestGenerateReport:
    def test_returns_string(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert isinstance(report, str)

    def test_contains_header(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "AnomalyShield Detection Report" in report

    def test_contains_summary_section(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "## Summary" in report

    def test_contains_per_detector_section(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "## Per-Detector Details" in report

    def test_contains_ensemble_section(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "## Ensemble Summary" in report

    def test_contains_detector_name(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "DetectorA" in report

    def test_contains_metrics_section_when_metrics_present(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_with_metrics())
        assert "## Metrics Comparison" in report

    def test_no_metrics_section_when_metrics_absent(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_no_metrics())
        assert "## Metrics Comparison" not in report

    def test_writes_to_file_when_output_path_given(self) -> None:
        from src.utils import generate_report

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            report = generate_report(_make_results_no_metrics(), output_path=tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                written = f.read()
            assert written == report
        finally:
            os.unlink(tmp_path)

    def test_file_content_matches_returned_string(self) -> None:
        from src.utils import generate_report

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            report = generate_report(_make_results_with_metrics(), output_path=tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                written = f.read()
            assert written == report
        finally:
            os.unlink(tmp_path)

    def test_multi_detector_report_contains_all_names(self) -> None:
        from src.utils import generate_report

        rng = np.random.default_rng(1)
        preds_a = np.ones(50, dtype=int)
        preds_a[:5] = -1
        preds_b = np.ones(50, dtype=int)
        preds_b[:8] = -1
        results = {
            "Alpha": {"predictions": preds_a, "scores": rng.uniform(0, 1, 50)},
            "Beta": {"predictions": preds_b, "scores": rng.uniform(0, 1, 50)},
        }
        report = generate_report(results)
        assert "Alpha" in report
        assert "Beta" in report

    def test_anomaly_count_in_report(self) -> None:
        from src.utils import generate_report

        preds = np.ones(100, dtype=int)
        preds[:15] = -1  # exactly 15 anomalies
        results = {
            "MyDet": {
                "predictions": preds,
                "scores": np.zeros(100),
            }
        }
        report = generate_report(results)
        assert "15" in report

    def test_report_with_metrics_contains_metric_values(self) -> None:
        from src.utils import generate_report

        report = generate_report(_make_results_with_metrics())
        # Accuracy value 0.9 should appear formatted
        assert "0.9000" in report

    def test_no_output_path_does_not_create_file(self) -> None:
        from src.utils import generate_report

        generate_report(_make_results_no_metrics(), output_path=None)
        # Verify no stray files were created in cwd
        assert not os.path.exists("None")
