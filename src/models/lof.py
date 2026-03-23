"""Local Outlier Factor anomaly detector for AnomalyShield."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.detector import BaseDetector


class LOFDetector(BaseDetector):
    """Anomaly detector based on sklearn's Local Outlier Factor.

    LOF measures the local deviation of density of a given sample with respect
    to its neighbors.  Points with substantially lower density than their
    neighbors are considered outliers.

    Parameters
    ----------
    name : str
        Detector name for identification in AnomalyShield.
    n_neighbors : int
        Number of neighbors to use for LOF computation.
    contamination : str | float
        Expected proportion of anomalies. ``"auto"`` uses the offset-based method.
    novelty : bool
        Must be ``True`` to allow calling ``predict`` on new, unseen data.
    metric : str
        Distance metric for neighbor queries.
    """

    def __init__(
        self,
        name: str = "LOF",
        n_neighbors: int = 20,
        contamination: str | float = "auto",
        novelty: bool = True,
        metric: str = "minkowski",
    ) -> None:
        super().__init__(
            name=name,
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty,
            metric=metric,
        )
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty,
            metric=metric,
        )

    def fit(self, X: np.ndarray) -> LOFDetector:
        """Fit the LOF model on training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        LOFDetector
            The fitted detector (self).
        """
        self._model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels using the fitted LOF model.

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

        Negates sklearn's ``score_samples`` output, which returns higher values
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
        return -self._model.score_samples(X)

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Detector '{self.name}' has not been fitted. Call fit() first."
            )
