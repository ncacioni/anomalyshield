"""External data source adapters — yfinance and PostgreSQL."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data.loader import TimeSeriesLoader


class YFinanceSource:
    """Fetches OHLCV data from Yahoo Finance via yfinance."""

    @staticmethod
    def fetch(
        ticker: str,
        start: str,
        end: str,
        column: str = "Close",
    ) -> pd.DataFrame:
        """Download historical price data for a ticker.

        Parameters
        ----------
        ticker:
            Yahoo Finance ticker symbol, e.g. ``"AAPL"``.
        start:
            Start date string in ISO-8601 format, e.g. ``"2023-01-01"``.
        end:
            End date string in ISO-8601 format, e.g. ``"2024-01-01"``.
        column:
            OHLCV column to extract.  Defaults to ``"Close"``.

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with DatetimeIndex and a single numeric
            column named after *column*.

        Raises
        ------
        ImportError
            If yfinance is not installed.
        ValueError
            If the requested column is not present or data is empty.
        """
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for YFinanceSource. "
                "Install it with: pip install yfinance"
            ) from exc

        raw: pd.DataFrame = yf.download(ticker, start=start, end=end, progress=False)

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for ticker '{ticker}' "
                f"between {start} and {end}."
            )

        # yfinance may return MultiIndex columns when auto_adjust=True
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if column not in raw.columns:
            raise ValueError(
                f"Column '{column}' not found in yfinance data. "
                f"Available columns: {list(raw.columns)}"
            )

        df = raw[[column]].copy()
        df.index.name = "date"
        df[column] = pd.to_numeric(df[column], errors="raise")
        df = df.sort_index()

        return df


class PostgreSQLSource:
    """Reads and writes time series data to a PostgreSQL database.

    All SQL uses parameterized queries via SQLAlchemy ``text()`` with
    bound parameters — string interpolation is never used.

    Expected schema
    ---------------
    time_series (
        id          BIGSERIAL PRIMARY KEY,
        name        TEXT NOT NULL,
        ts          TIMESTAMPTZ NOT NULL,
        value       DOUBLE PRECISION NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )

    detection_results (
        id           BIGSERIAL PRIMARY KEY,
        series_name  TEXT NOT NULL,
        method       TEXT NOT NULL,
        results      JSONB NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize the source with a SQLAlchemy connection string.

        Parameters
        ----------
        connection_string:
            A full SQLAlchemy URL, e.g.
            ``"postgresql+psycopg2://user:pass@host:5432/dbname"``.
        """
        self._engine: Engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_series(self, series_name: str) -> pd.DataFrame:
        """Read a named time series from the ``time_series`` table.

        Parameters
        ----------
        series_name:
            The ``name`` column value used to identify the series.

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with DatetimeIndex and a ``value`` column.

        Raises
        ------
        ValueError
            If no rows are found for the given series name.
        """
        query = text(
            "SELECT ts AS date, value "
            "FROM time_series "
            "WHERE name = :name "
            "ORDER BY ts ASC"
        )

        with self._engine.connect() as conn:
            result = conn.execute(query, {"name": series_name})
            rows = result.fetchall()

        if not rows:
            raise ValueError(
                f"No data found in time_series for series_name='{series_name}'."
            )

        df = pd.DataFrame(rows, columns=["date", "value"])
        return TimeSeriesLoader.from_dataframe(df, date_col="date", value_col="value")

    def save_series(self, name: str, df: pd.DataFrame) -> None:
        """Persist a time series DataFrame to the ``time_series`` table.

        Existing rows for *name* are NOT deleted — use this method to
        append new data.  Duplicate (name, ts) rows will raise a DB-level
        unique constraint violation if one is defined on the table.

        Parameters
        ----------
        name:
            Logical name for the series (stored in the ``name`` column).
        df:
            DataFrame with a DatetimeIndex and a ``value`` column.

        Raises
        ------
        ValueError
            If *df* does not have a DatetimeIndex or lacks a ``value`` column.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex.")
        if "value" not in df.columns:
            raise ValueError("df must have a 'value' column.")

        insert_stmt = text(
            "INSERT INTO time_series (name, ts, value) "
            "VALUES (:name, :ts, :value)"
        )

        rows = [
            {"name": name, "ts": ts, "value": float(row["value"])}
            for ts, row in df.iterrows()
        ]

        with self._engine.begin() as conn:
            conn.execute(insert_stmt, rows)

    def save_results(
        self,
        series_name: str,
        method: str,
        results: dict,
    ) -> None:
        """Persist anomaly detection results to the ``detection_results`` table.

        Parameters
        ----------
        series_name:
            Name of the time series the results relate to.
        method:
            Identifier of the detection method used, e.g. ``"zscore"``.
        results:
            Dictionary of results — will be serialized to JSONB.
        """
        import json  # noqa: PLC0415

        insert_stmt = text(
            "INSERT INTO detection_results (series_name, method, results) "
            "VALUES (:series_name, :method, :results)"
        )

        with self._engine.begin() as conn:
            conn.execute(
                insert_stmt,
                {
                    "series_name": series_name,
                    "method": method,
                    "results": json.dumps(results),
                },
            )
