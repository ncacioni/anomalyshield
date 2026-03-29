"""Isolation Forest anomaly detector for AnomalyShield."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from src.detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Anomaly detector based on sklearn's Isolation Forest.

    Isolation Forest isolates anomalies by randomly selecting a feature and then
    randomly selecting a split value between the max and min of the selected feature.
    Anomalies require fewer splits to isolate, resulting in shorter path lengths.

    Parameters
    ----------
    name : str
        Detector name for identification in AnomalyShield.
    n_estimators : int
        Number of base estimators (trees) in the ensemble.
    contamination : str | float
        Expected proportion of anomalies. ``"auto"`` uses the offset-based method.
    max_features : float
        Fraction of features to draw for each tree.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "IsolationForest",
        n_estimators: int = 100,
        contamination: str | float = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
        )
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        """Fit the Isolation Forest on training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        IsolationForestDetector
            The fitted detector (self).
        """
        self._model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels using the fitted Isolation Forest.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Labels: -1 for anomaly, 1 for normal.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        self._check_is_fitted()
        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous).

        Negates sklearn's ``decision_function`` output, which returns higher values
        for normal samples, so that our convention (higher = more anomalous) holds.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Anomaly scores where higher values indicate more anomalous samples.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        self._check_is_fitted()
        return -self._model.decision_function(X)

