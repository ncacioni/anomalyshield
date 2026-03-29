"""Core detector framework for AnomalyShield."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.utils import comparison_table, evaluate_detector


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors.

    Subclasses must implement ``fit``, ``predict``, and ``score_samples``.
    The label convention follows sklearn: -1 for anomaly, 1 for normal.
    """

    def __init__(self, name: str, **params: object) -> None:
        self.name = name
        self.params = params
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseDetector:
        """Fit the detector on training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        BaseDetector
            The fitted detector (self).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Labels: -1 for anomaly, 1 for normal.
        """
        ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Anomaly scores where higher values indicate more anomalous samples.
        """
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return predictions.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Labels: -1 for anomaly, 1 for normal.
        """
        return self.fit(X).predict(X)

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Detector '{self.name}' has not been fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class AnomalyShield:
    """Orchestrator that runs multiple detectors, compares results, and builds ensembles.

    Examples
    --------
    >>> shield = AnomalyShield()
    >>> shield.add_detector(some_detector)
    >>> results = shield.run_all(X, y_true=y)
    >>> shield.compare()
    """

    def __init__(self) -> None:
        self.detectors: dict[str, BaseDetector] = {}
        self.results: dict[str, dict] = {}

    def add_detector(self, detector: BaseDetector) -> None:
        """Register a detector.

        Parameters
        ----------
        detector : BaseDetector
            Detector instance to add. Its ``name`` attribute is used as the key.

        Raises
        ------
        ValueError
            If a detector with the same name is already registered.
        """
        if detector.name in self.detectors:
            raise ValueError(
                f"Detector with name '{detector.name}' already registered. "
                "Use a unique name for each detector."
            )
        self.detectors[detector.name] = detector

    def run_all(
        self,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> dict[str, dict]:
        """Run all registered detectors on the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y_true : np.ndarray | None
            Ground truth labels (-1/1 or 0/1). If provided, evaluation metrics
            are computed for each detector.

        Returns
        -------
        dict[str, dict]
            Results keyed by detector name. Each value is a dict with:
            - ``predictions``: np.ndarray of -1/1 labels
            - ``scores``: np.ndarray of anomaly scores
            - ``metrics``: dict of evaluation metrics (only if y_true provided)
        """
        if not self.detectors:
            raise ValueError("No detectors registered. Use add_detector() first.")

        self.results = {}

        for name, detector in self.detectors.items():
            predictions = detector.fit_predict(X)
            scores = detector.score_samples(X)

            result: dict = {
                "predictions": predictions,
                "scores": scores,
            }

            if y_true is not None:
                result["metrics"] = evaluate_detector(y_true, predictions, scores)

            self.results[name] = result

        return self.results

    def compare(self) -> pd.DataFrame:
        """Return a comparison table of all detectors' metrics.

        Only works if ``y_true`` was provided to ``run_all``.

        Returns
        -------
        pd.DataFrame
            DataFrame with detector names as rows and metrics as columns.

        Raises
        ------
        ValueError
            If no results exist or no metrics were computed.
        """
        if not self.results:
            raise ValueError("No results available. Call run_all() first.")
        return comparison_table(self.results)

    def get_ensemble_predictions(self, strategy: str = "majority") -> np.ndarray:
        """Combine predictions from all detectors using the given strategy.

        Parameters
        ----------
        strategy : str
            Ensemble strategy. One of:
            - ``"majority"``: anomaly if more than half of detectors flag it
            - ``"unanimous"``: anomaly only if ALL detectors flag it
            - ``"any"``: anomaly if ANY detector flags it

        Returns
        -------
        np.ndarray
            Ensemble labels: -1 for anomaly, 1 for normal.

        Raises
        ------
        ValueError
            If no results exist or strategy is unknown.
        """
        if not self.results:
            raise ValueError("No results available. Call run_all() first.")

        valid_strategies = {"majority", "unanimous", "any"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {sorted(valid_strategies)}"
            )

        all_preds = np.array(
            [result["predictions"] for result in self.results.values()]
        )
        # Count how many detectors flagged each sample as anomaly (-1)
        anomaly_votes = np.sum(all_preds == -1, axis=0)
        n_detectors = len(self.results)

        if strategy == "majority":
            ensemble = np.where(anomaly_votes > n_detectors / 2, -1, 1)
        elif strategy == "unanimous":
            ensemble = np.where(anomaly_votes == n_detectors, -1, 1)
        elif strategy == "any":
            ensemble = np.where(anomaly_votes > 0, -1, 1)

        return ensemble
