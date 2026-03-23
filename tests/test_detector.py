"""Tests for the detector framework: IsolationForestDetector, AnomalyShield, and utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.detector import AnomalyShield, BaseDetector
from src.models.isolation_forest import IsolationForestDetector
from src.utils import comparison_table, evaluate_detector, set_random_seeds


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def normal_data() -> np.ndarray:
    """200 samples from a tight 2D Gaussian cluster (mostly normal)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(200, 2))


@pytest.fixture()
def anomaly_data() -> np.ndarray:
    """20 samples far from origin (anomalous)."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=10.0, scale=0.5, size=(20, 2))


@pytest.fixture()
def mixed_data(normal_data: np.ndarray, anomaly_data: np.ndarray) -> np.ndarray:
    """220 samples: 200 normal + 20 anomalous, stacked."""
    return np.vstack([normal_data, anomaly_data])


@pytest.fixture()
def y_true_mixed(normal_data: np.ndarray, anomaly_data: np.ndarray) -> np.ndarray:
    """Ground truth labels: 1 for normal, -1 for anomaly."""
    normal_labels = np.ones(len(normal_data), dtype=int)
    anomaly_labels = -np.ones(len(anomaly_data), dtype=int)
    return np.concatenate([normal_labels, anomaly_labels])


@pytest.fixture()
def fitted_detector(mixed_data: np.ndarray) -> IsolationForestDetector:
    det = IsolationForestDetector(name="IF_test", contamination=0.1, random_state=42)
    det.fit(mixed_data)
    return det


# ---------------------------------------------------------------------------
# IsolationForestDetector — construction
# ---------------------------------------------------------------------------


class TestIsolationForestDetectorInit:
    def test_name_stored(self) -> None:
        det = IsolationForestDetector(name="MyIF")
        assert det.name == "MyIF"

    def test_default_name(self) -> None:
        det = IsolationForestDetector()
        assert det.name == "IsolationForest"

    def test_not_fitted_on_init(self) -> None:
        det = IsolationForestDetector()
        assert det.is_fitted is False

    def test_params_stored(self) -> None:
        det = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=7)
        assert det.params["n_estimators"] == 50
        assert det.params["contamination"] == 0.05
        assert det.params["random_state"] == 7


# ---------------------------------------------------------------------------
# IsolationForestDetector — fit
# ---------------------------------------------------------------------------


class TestIsolationForestDetectorFit:
    def test_returns_self(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector(random_state=0)
        result = det.fit(mixed_data)
        assert result is det

    def test_is_fitted_true_after_fit(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector(random_state=0)
        det.fit(mixed_data)
        assert det.is_fitted is True


# ---------------------------------------------------------------------------
# IsolationForestDetector — predict
# ---------------------------------------------------------------------------


class TestIsolationForestDetectorPredict:
    def test_output_shape(self, fitted_detector: IsolationForestDetector, mixed_data: np.ndarray) -> None:
        preds = fitted_detector.predict(mixed_data)
        assert preds.shape == (len(mixed_data),)

    def test_labels_are_minus1_or_1(self, fitted_detector: IsolationForestDetector, mixed_data: np.ndarray) -> None:
        preds = fitted_detector.predict(mixed_data)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_predict_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict(mixed_data)

    def test_anomaly_cluster_mostly_flagged(
        self, fitted_detector: IsolationForestDetector, anomaly_data: np.ndarray
    ) -> None:
        preds = fitted_detector.predict(anomaly_data)
        # Anomalous cluster at (10,10) should be largely detected
        anomaly_rate = np.mean(preds == -1)
        assert anomaly_rate > 0.5

    def test_normal_cluster_mostly_normal(
        self, fitted_detector: IsolationForestDetector, normal_data: np.ndarray
    ) -> None:
        preds = fitted_detector.predict(normal_data)
        normal_rate = np.mean(preds == 1)
        assert normal_rate > 0.85


# ---------------------------------------------------------------------------
# IsolationForestDetector — score_samples
# ---------------------------------------------------------------------------


class TestIsolationForestDetectorScoreSamples:
    def test_output_shape(self, fitted_detector: IsolationForestDetector, mixed_data: np.ndarray) -> None:
        scores = fitted_detector.score_samples(mixed_data)
        assert scores.shape == (len(mixed_data),)

    def test_anomalies_have_higher_scores(
        self,
        fitted_detector: IsolationForestDetector,
        normal_data: np.ndarray,
        anomaly_data: np.ndarray,
    ) -> None:
        normal_scores = fitted_detector.score_samples(normal_data)
        anomaly_scores = fitted_detector.score_samples(anomaly_data)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_score_before_fit_raises(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.score_samples(mixed_data)


# ---------------------------------------------------------------------------
# IsolationForestDetector — reproducibility
# ---------------------------------------------------------------------------


class TestIsolationForestDetectorReproducibility:
    def test_same_seed_same_predictions(self, mixed_data: np.ndarray) -> None:
        det1 = IsolationForestDetector(random_state=42)
        det2 = IsolationForestDetector(random_state=42)
        preds1 = det1.fit(mixed_data).predict(mixed_data)
        preds2 = det2.fit(mixed_data).predict(mixed_data)
        np.testing.assert_array_equal(preds1, preds2)

    def test_same_seed_same_scores(self, mixed_data: np.ndarray) -> None:
        det1 = IsolationForestDetector(random_state=0)
        det2 = IsolationForestDetector(random_state=0)
        det1.fit(mixed_data)
        det2.fit(mixed_data)
        np.testing.assert_array_equal(
            det1.score_samples(mixed_data), det2.score_samples(mixed_data)
        )

    def test_different_seeds_may_differ(self, mixed_data: np.ndarray) -> None:
        det1 = IsolationForestDetector(random_state=1)
        det2 = IsolationForestDetector(random_state=2)
        scores1 = det1.fit(mixed_data).score_samples(mixed_data)
        scores2 = det2.fit(mixed_data).score_samples(mixed_data)
        # With only 2 different seeds scores *might* coincide by chance, but very unlikely
        assert not np.array_equal(scores1, scores2)


# ---------------------------------------------------------------------------
# IsolationForestDetector — fit_predict (inherited)
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_fit_predict_returns_labels(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector(random_state=42)
        preds = det.fit_predict(mixed_data)
        assert preds.shape == (len(mixed_data),)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_fit_predict_sets_is_fitted(self, mixed_data: np.ndarray) -> None:
        det = IsolationForestDetector(random_state=42)
        det.fit_predict(mixed_data)
        assert det.is_fitted is True


# ---------------------------------------------------------------------------
# AnomalyShield — add_detector
# ---------------------------------------------------------------------------


class TestAnomalyShieldAddDetector:
    def test_add_detector_registered(self) -> None:
        shield = AnomalyShield()
        det = IsolationForestDetector(name="IF1")
        shield.add_detector(det)
        assert "IF1" in shield.detectors

    def test_duplicate_name_raises(self) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF1"))
        with pytest.raises(ValueError, match="already registered"):
            shield.add_detector(IsolationForestDetector(name="IF1"))

    def test_multiple_detectors(self) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF1"))
        shield.add_detector(IsolationForestDetector(name="IF2"))
        assert len(shield.detectors) == 2


# ---------------------------------------------------------------------------
# AnomalyShield — run_all
# ---------------------------------------------------------------------------


class TestAnomalyShieldRunAll:
    def test_no_detectors_raises(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        with pytest.raises(ValueError, match="No detectors"):
            shield.run_all(mixed_data)

    def test_run_all_returns_dict(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", random_state=42))
        results = shield.run_all(mixed_data)
        assert isinstance(results, dict)
        assert "IF" in results

    def test_result_has_predictions_and_scores(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", random_state=42))
        results = shield.run_all(mixed_data)
        assert "predictions" in results["IF"]
        assert "scores" in results["IF"]

    def test_result_no_metrics_without_y_true(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", random_state=42))
        results = shield.run_all(mixed_data)
        assert "metrics" not in results["IF"]

    def test_result_has_metrics_with_y_true(
        self, mixed_data: np.ndarray, y_true_mixed: np.ndarray
    ) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", contamination=0.1, random_state=42))
        results = shield.run_all(mixed_data, y_true=y_true_mixed)
        assert "metrics" in results["IF"]

    def test_run_all_multiple_detectors(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF1", random_state=1))
        shield.add_detector(IsolationForestDetector(name="IF2", random_state=2))
        results = shield.run_all(mixed_data)
        assert set(results.keys()) == {"IF1", "IF2"}


# ---------------------------------------------------------------------------
# AnomalyShield — compare
# ---------------------------------------------------------------------------


class TestAnomalyShieldCompare:
    def test_compare_returns_dataframe(
        self, mixed_data: np.ndarray, y_true_mixed: np.ndarray
    ) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", contamination=0.1, random_state=42))
        shield.run_all(mixed_data, y_true=y_true_mixed)
        table = shield.compare()
        assert isinstance(table, pd.DataFrame)

    def test_compare_index_is_detector_names(
        self, mixed_data: np.ndarray, y_true_mixed: np.ndarray
    ) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", contamination=0.1, random_state=42))
        shield.run_all(mixed_data, y_true=y_true_mixed)
        table = shield.compare()
        assert "IF" in table.index

    def test_compare_contains_standard_metrics(
        self, mixed_data: np.ndarray, y_true_mixed: np.ndarray
    ) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", contamination=0.1, random_state=42))
        shield.run_all(mixed_data, y_true=y_true_mixed)
        table = shield.compare()
        for col in ("accuracy", "precision", "recall", "f1"):
            assert col in table.columns

    def test_compare_before_run_all_raises(self) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF"))
        with pytest.raises(ValueError):
            shield.compare()

    def test_compare_without_y_true_raises(self, mixed_data: np.ndarray) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF", random_state=42))
        shield.run_all(mixed_data)  # no y_true
        with pytest.raises(ValueError, match="No metrics"):
            shield.compare()

    def test_compare_multi_detector_rows(
        self, mixed_data: np.ndarray, y_true_mixed: np.ndarray
    ) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF1", contamination=0.1, random_state=1))
        shield.add_detector(IsolationForestDetector(name="IF2", contamination=0.1, random_state=2))
        shield.run_all(mixed_data, y_true=y_true_mixed)
        table = shield.compare()
        assert len(table) == 2


# ---------------------------------------------------------------------------
# AnomalyShield — get_ensemble_predictions
# ---------------------------------------------------------------------------


class TestEnsemblePredictions:
    def _shield_with_results(self, mixed_data: np.ndarray) -> AnomalyShield:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF1", contamination=0.1, random_state=1))
        shield.add_detector(IsolationForestDetector(name="IF2", contamination=0.1, random_state=2))
        shield.add_detector(IsolationForestDetector(name="IF3", contamination=0.1, random_state=3))
        shield.run_all(mixed_data)
        return shield

    def test_majority_output_shape(self, mixed_data: np.ndarray) -> None:
        shield = self._shield_with_results(mixed_data)
        preds = shield.get_ensemble_predictions(strategy="majority")
        assert preds.shape == (len(mixed_data),)

    def test_majority_labels_are_minus1_or_1(self, mixed_data: np.ndarray) -> None:
        shield = self._shield_with_results(mixed_data)
        preds = shield.get_ensemble_predictions(strategy="majority")
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_unanimous_output_shape(self, mixed_data: np.ndarray) -> None:
        shield = self._shield_with_results(mixed_data)
        preds = shield.get_ensemble_predictions(strategy="unanimous")
        assert preds.shape == (len(mixed_data),)

    def test_any_output_shape(self, mixed_data: np.ndarray) -> None:
        shield = self._shield_with_results(mixed_data)
        preds = shield.get_ensemble_predictions(strategy="any")
        assert preds.shape == (len(mixed_data),)

    def test_unanimous_le_majority_anomalies(self, mixed_data: np.ndarray) -> None:
        """'unanimous' flags fewer or equal anomalies than 'majority'."""
        shield = self._shield_with_results(mixed_data)
        majority = shield.get_ensemble_predictions(strategy="majority")
        unanimous = shield.get_ensemble_predictions(strategy="unanimous")
        assert np.sum(unanimous == -1) <= np.sum(majority == -1)

    def test_any_ge_majority_anomalies(self, mixed_data: np.ndarray) -> None:
        """'any' flags more or equal anomalies than 'majority'."""
        shield = self._shield_with_results(mixed_data)
        majority = shield.get_ensemble_predictions(strategy="majority")
        any_preds = shield.get_ensemble_predictions(strategy="any")
        assert np.sum(any_preds == -1) >= np.sum(majority == -1)

    def test_unknown_strategy_raises(self, mixed_data: np.ndarray) -> None:
        shield = self._shield_with_results(mixed_data)
        with pytest.raises(ValueError, match="Unknown strategy"):
            shield.get_ensemble_predictions(strategy="weighted")

    def test_no_results_raises(self) -> None:
        shield = AnomalyShield()
        shield.add_detector(IsolationForestDetector(name="IF"))
        with pytest.raises(ValueError, match="No results"):
            shield.get_ensemble_predictions()


# ---------------------------------------------------------------------------
# evaluate_detector
# ---------------------------------------------------------------------------


class TestEvaluateDetector:
    def test_returns_dict(self) -> None:
        y_true = np.array([-1, 1, 1, -1, 1])
        y_pred = np.array([-1, 1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred)
        assert isinstance(result, dict)

    def test_keys_present(self) -> None:
        y_true = np.array([-1, 1, 1, -1, 1])
        y_pred = np.array([-1, 1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in result

    def test_perfect_predictions(self) -> None:
        y_true = np.array([-1, -1, 1, 1, 1])
        y_pred = np.array([-1, -1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self) -> None:
        y_true = np.array([-1, -1, 1, 1])
        y_pred = np.array([1, 1, -1, -1])
        result = evaluate_detector(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_accepts_0_1_labels(self) -> None:
        # 0 = anomaly, 1 = normal in the 0/1 convention from generate_synthetic
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([-1, -1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_auc_roc_included_when_scores_given(self) -> None:
        y_true = np.array([-1, -1, 1, 1, 1])
        y_pred = np.array([-1, -1, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.15])
        result = evaluate_detector(y_true, y_pred, y_scores=scores)
        assert "auc_roc" in result

    def test_auc_roc_not_included_without_scores(self) -> None:
        y_true = np.array([-1, -1, 1, 1, 1])
        y_pred = np.array([-1, -1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred)
        assert "auc_roc" not in result

    def test_metric_values_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.choice([-1, 1], size=100)
        y_pred = rng.choice([-1, 1], size=100)
        result = evaluate_detector(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------


class TestComparisonTable:
    def test_returns_dataframe(self) -> None:
        results = {
            "detA": {"predictions": np.array([1, -1]), "scores": np.array([0.1, 0.9]),
                     "metrics": {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}},
        }
        table = comparison_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_index_is_detector_name(self) -> None:
        results = {
            "detA": {"predictions": np.array([1, -1]), "scores": np.array([0.1, 0.9]),
                     "metrics": {"accuracy": 0.9, "f1": 0.75}},
        }
        table = comparison_table(results)
        assert "detA" in table.index

    def test_index_name_is_detector(self) -> None:
        results = {
            "detA": {"metrics": {"accuracy": 0.9}},
        }
        table = comparison_table(results)
        assert table.index.name == "detector"

    def test_columns_match_metric_keys(self) -> None:
        metrics = {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}
        results = {"detA": {"metrics": metrics}}
        table = comparison_table(results)
        for col in metrics:
            assert col in table.columns

    def test_multiple_detectors_multiple_rows(self) -> None:
        results = {
            "detA": {"metrics": {"accuracy": 0.9, "f1": 0.8}},
            "detB": {"metrics": {"accuracy": 0.7, "f1": 0.6}},
        }
        table = comparison_table(results)
        assert len(table) == 2

    def test_no_metrics_raises(self) -> None:
        results = {
            "detA": {"predictions": np.array([1, -1]), "scores": np.array([0.1, 0.9])},
        }
        with pytest.raises(ValueError, match="No metrics"):
            comparison_table(results)

    def test_empty_results_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            comparison_table({})


# ---------------------------------------------------------------------------
# set_random_seeds
# ---------------------------------------------------------------------------


class TestSetRandomSeeds:
    def test_numpy_reproducibility(self) -> None:
        set_random_seeds(42)
        arr1 = np.random.rand(10)
        set_random_seeds(42)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_differ(self) -> None:
        set_random_seeds(1)
        arr1 = np.random.rand(10)
        set_random_seeds(2)
        arr2 = np.random.rand(10)
        assert not np.array_equal(arr1, arr2)

    def test_does_not_raise(self) -> None:
        # Should not raise even if torch is not installed
        set_random_seeds(0)
        set_random_seeds(100)
