"""Elliptic Envelope anomaly detector for AnomalyShield."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope

from src.detector import BaseDetector


class EllipticEnvelopeDetector(BaseDetector):
    """Anomaly detector based on sklearn's Elliptic Envelope.

    Fits a robust covariance estimate to the data and flags samples that lie
    far from the fitted ellipsoid as anomalies.  Assumes the inlier data is
    Gaussian-distributed.

    Parameters
    ----------
    name : str
        Detector name for identification in AnomalyShield.
    contamination : float
        Expected proportion of anomalies in the training data.
    support_fraction : float | None
        Fraction of points to include in the raw MCD estimate.  ``None``
        lets sklearn choose automatically.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "EllipticEnvelope",
        contamination: float = 0.1,
        support_fraction: float | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=random_state,
        )
        self._model = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> EllipticEnvelopeDetector:
        """Fit the Elliptic Envelope on training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        EllipticEnvelopeDetector
            The fitted detector (self).
        """
        self._model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels using the fitted Elliptic Envelope.

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

        Negates sklearn's ``decision_function`` output, which returns higher
        values for normal samples, so that our convention (higher = more
        anomalous) holds.

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

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Detector '{self.name}' has not been fitted. Call fit() first."
            )
