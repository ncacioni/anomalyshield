"""Prophet-based time series forecaster and anomaly detector for AnomalyShield."""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet


class ProphetForecaster:
    """Time series forecaster and anomaly detector using Meta's Prophet.

    Unlike the other detectors in AnomalyShield, Prophet operates on
    timestamped DataFrames rather than numpy arrays and does NOT subclass
    :class:`BaseDetector`.

    Parameters
    ----------
    changepoint_prior_scale : float
        Regularization for trend changepoints.  Larger values allow more
        flexibility.
    seasonality_mode : str
        ``"additive"`` or ``"multiplicative"`` seasonality.
    interval_width : float
        Width of the uncertainty interval (e.g. 0.95 for 95%).
    yearly_seasonality : bool | str | int
        Whether to include yearly seasonality.
    weekly_seasonality : bool | str | int
        Whether to include weekly seasonality.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        interval_width: float = 0.95,
        yearly_seasonality: bool | str | int = True,
        weekly_seasonality: bool | str | int = True,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality

        self._model: Prophet | None = None
        self.is_fitted: bool = False

    def _make_model(self) -> Prophet:
        """Create a fresh Prophet instance with the configured parameters."""
        # Suppress Prophet's verbose cmdstanpy / pystan logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        return Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
        )

    @staticmethod
    def _to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert a DataFrame with DatetimeIndex and 'value' column to Prophet format."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "Input DataFrame must have a DatetimeIndex. "
                "Set the datetime column as the index before calling this method."
            )
        if "value" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain a 'value' column."
            )
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df["value"].values,
        })
        return prophet_df

    def fit(self, df: pd.DataFrame) -> ProphetForecaster:
        """Fit Prophet on the provided time series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and a ``value`` column.

        Returns
        -------
        ProphetForecaster
            The fitted forecaster (self).
        """
        prophet_df = self._to_prophet_df(df)
        self._model = self._make_model()
        self._model.fit(prophet_df)
        self.is_fitted = True
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """Generate a forecast for future periods.

        Parameters
        ----------
        periods : int
            Number of future time steps to forecast.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with columns: ``ds``, ``yhat``,
            ``yhat_lower``, ``yhat_upper``.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted yet.
        """
        self._check_is_fitted()
        future = self._model.make_future_dataframe(periods=periods)  # type: ignore[union-attr]
        forecast = self._model.predict(future)  # type: ignore[union-attr]
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and flag points outside the prediction interval as anomalies.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and a ``value`` column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns: ``yhat``, ``yhat_lower``,
            ``yhat_upper``, ``is_anomaly``.
        """
        self.fit(df)

        prophet_df = self._to_prophet_df(df)
        forecast = self._model.predict(prophet_df)  # type: ignore[union-attr]

        result = df.copy()
        result["yhat"] = forecast["yhat"].values
        result["yhat_lower"] = forecast["yhat_lower"].values
        result["yhat_upper"] = forecast["yhat_upper"].values
        result["is_anomaly"] = (
            (result["value"] < result["yhat_lower"])
            | (result["value"] > result["yhat_upper"])
        )
        return result

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                "ProphetForecaster has not been fitted. Call fit() first."
            )
