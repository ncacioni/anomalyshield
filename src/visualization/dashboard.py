"""Reusable Streamlit component functions for the AnomalyShield dashboard."""

from __future__ import annotations

import io
import os
import sys
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so src imports work regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar_config() -> dict:
    """Render the sidebar and return a configuration dictionary.

    Returns
    -------
    dict
        Keys:
        - data_source: str — "Sample Data" | "Upload CSV" | "Yahoo Finance"
        - uploaded_file: UploadedFile | None
        - ticker: str | None
        - yf_start: date | None
        - yf_end: date | None
        - models: list[str] — selected detector names
        - use_prophet: bool
        - contamination: float
        - ae_epochs: int
    """
    with st.sidebar:
        st.title("AnomalyShield")
        st.caption("Anomaly detection suite")
        st.divider()

        # --- Data source ---
        st.subheader("Data Source")
        data_source = st.radio(
            "Select source",
            options=["Sample Data", "Upload CSV", "Yahoo Finance"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        ticker = None
        yf_start = None
        yf_end = None

        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="CSV must contain a 'date' column and a 'value' column.",
            )

        elif data_source == "Yahoo Finance":
            ticker = st.text_input(
                "Ticker symbol",
                value="AAPL",
                placeholder="e.g. AAPL, MSFT, BTC-USD",
            ).strip().upper()
            col1, col2 = st.columns(2)
            with col1:
                yf_start = st.date_input(
                    "Start date",
                    value=date.today() - timedelta(days=730),
                    max_value=date.today() - timedelta(days=1),
                )
            with col2:
                yf_end = st.date_input(
                    "End date",
                    value=date.today(),
                    max_value=date.today(),
                )

        st.divider()

        # --- Model selection ---
        st.subheader("Detectors")

        _all_models = [
            "Isolation Forest",
            "LOF",
            "Elliptic Envelope",
            "Autoencoder",
        ]
        models = st.multiselect(
            "Select detectors",
            options=_all_models,
            default=["Isolation Forest", "LOF"],
            help="Choose one or more anomaly detection algorithms.",
        )

        use_prophet = st.toggle(
            "Enable Prophet forecasting",
            value=False,
            help="Prophet fits a trend + seasonality model and flags residual outliers.",
        )

        st.divider()

        # --- Model parameters ---
        st.subheader("Parameters")

        contamination = st.slider(
            "Contamination",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Expected fraction of anomalies in the data.",
        )

        ae_epochs = st.slider(
            "Autoencoder epochs",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Training epochs for the LSTM Autoencoder (only used when Autoencoder is selected).",
        )

        st.divider()
        st.caption("AnomalyShield v1.0")

    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "ticker": ticker,
        "yf_start": yf_start,
        "yf_end": yf_end,
        "models": models,
        "use_prophet": use_prophet,
        "contamination": contamination,
        "ae_epochs": ae_epochs,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: dict) -> Optional[pd.DataFrame]:
    """Load and return a standardized DataFrame based on the sidebar config.

    Parameters
    ----------
    config : dict
        Configuration dict returned by :func:`sidebar_config`.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame with a DatetimeIndex, at minimum a ``value`` column, and
        optionally an ``is_anomaly`` column.  Returns ``None`` on any error.
    """
    source = config["data_source"]

    try:
        if source == "Sample Data":
            return _load_sample_data()

        elif source == "Upload CSV":
            return _load_uploaded_csv(config["uploaded_file"])

        elif source == "Yahoo Finance":
            return _load_yahoo_finance(
                config["ticker"],
                config["yf_start"],
                config["yf_end"],
            )

    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load data: {exc}")

    return None


def _load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset, falling back to synthetic generation."""
    try:
        # Try the pre-built CSV first
        assets_csv = os.path.join(_PROJECT_ROOT, "assets", "sample_data.csv")
        if os.path.isfile(assets_csv):
            from src.data.loader import TimeSeriesLoader  # noqa: PLC0415
            df = TimeSeriesLoader.from_csv(assets_csv, date_col="date", value_col="value")
            # Re-attach is_anomaly if present in the raw file
            raw = pd.read_csv(assets_csv, parse_dates=["date"])
            if "is_anomaly" in raw.columns:
                raw = raw.set_index(pd.to_datetime(raw["date"])).sort_index()
                df["is_anomaly"] = raw["is_anomaly"].values
            return df
    except Exception:  # noqa: BLE001
        pass

    # Fall back to synthetic generation
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "datasets", os.path.join(_PROJECT_ROOT, "data", "datasets.py")
    )
    datasets_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(datasets_mod)  # type: ignore[union-attr]
    return datasets_mod.generate_synthetic()


def _load_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse an uploaded CSV file."""
    if uploaded_file is None:
        st.info("Upload a CSV file to get started.")
        return None

    raw = pd.read_csv(uploaded_file)

    # Auto-detect date column
    date_col = None
    for candidate in ("date", "Date", "datetime", "Datetime", "timestamp", "Timestamp", "ds"):
        if candidate in raw.columns:
            date_col = candidate
            break
    if date_col is None:
        # Try first column
        date_col = raw.columns[0]

    # Auto-detect value column
    value_col = None
    for candidate in ("value", "Value", "close", "Close", "y"):
        if candidate in raw.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_cols = raw.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric column found in CSV.")
        value_col = numeric_cols[0]

    from src.data.loader import TimeSeriesLoader  # noqa: PLC0415
    df = TimeSeriesLoader.from_csv(
        # Write to temp bytes buffer the loader can accept as path
        _write_temp_csv(uploaded_file, raw),
        date_col=date_col,
        value_col=value_col,
    )

    # Preserve is_anomaly if present
    if "is_anomaly" in raw.columns:
        raw_idx = pd.to_datetime(raw[date_col])
        raw = raw.set_index(raw_idx).sort_index()
        df["is_anomaly"] = raw["is_anomaly"].values[: len(df)]

    return df


def _write_temp_csv(uploaded_file, raw_df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp file and return its path."""
    import tempfile  # noqa: PLC0415

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    raw_df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _load_yahoo_finance(
    ticker: Optional[str],
    start: Optional[date],
    end: Optional[date],
) -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance."""
    if not ticker:
        st.info("Enter a ticker symbol to fetch data.")
        return None
    if start is None or end is None:
        st.info("Select start and end dates.")
        return None
    if start >= end:
        st.error("Start date must be before end date.")
        return None

    from src.data.sources import YFinanceSource  # noqa: PLC0415

    with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
        df = YFinanceSource.fetch(
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )

    return df


# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------

def show_data_overview(df: pd.DataFrame) -> None:
    """Display a concise overview: shape, statistics, and a time series plot.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized time series DataFrame with DatetimeIndex.
    """
    if df is None or df.empty:
        st.warning("No data to display.")
        return

    value_col = _get_value_col(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        date_range = (df.index.max() - df.index.min()).days
        st.metric("Date range", f"{date_range} days")
    with col3:
        missing = df[value_col].isna().sum()
        st.metric("Missing values", missing)
    with col4:
        if "is_anomaly" in df.columns:
            n_anomalies = int(df["is_anomaly"].sum())
            pct = n_anomalies / len(df) * 100
            st.metric("Labelled anomalies", f"{n_anomalies} ({pct:.1f}%)")
        else:
            st.metric("Ground truth", "Not available")

    st.markdown("#### Time Series")
    fig = _plot_raw_series(df, value_col)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Descriptive statistics"):
        st.dataframe(df[value_col].describe().to_frame().T, use_container_width=True)

    with st.expander("First 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)


def _get_value_col(df: pd.DataFrame) -> str:
    """Return the primary value column name."""
    for candidate in ("value", "Close", "close"):
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_label = [c for c in numeric_cols if c != "is_anomaly"]
    return non_label[0] if non_label else numeric_cols[0]


def _plot_raw_series(df: pd.DataFrame, value_col: str) -> go.Figure:
    """Build a clean Plotly time series figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[value_col],
            mode="lines",
            name=value_col,
            line={"color": "#4C9BE8", "width": 1.5},
        )
    )

    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies[value_col],
                    mode="markers",
                    name="Labelled anomaly",
                    marker={"color": "#FF4B4B", "size": 8, "symbol": "x"},
                )
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        hovermode="x unified",
        template="plotly_white",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

def run_detection(
    df: pd.DataFrame,
    config: dict,
) -> tuple[dict, Optional[np.ndarray]]:
    """Run the selected detectors and return results plus optional ground truth.

    Results are cached in ``st.session_state`` under the key
    ``"detection_results"`` so re-renders do not re-train models.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized time series DataFrame.
    config : dict
        Configuration dict from :func:`sidebar_config`.

    Returns
    -------
    tuple[dict, np.ndarray | None]
        ``(results_dict, y_true)`` where ``results_dict`` maps detector name
        to a dict with keys ``predictions``, ``scores``, and optionally
        ``metrics``.  ``y_true`` is an int array of -1/1 labels or ``None``
        when no ground truth exists.
    """
    # Build a cache key from the parameters that affect the result
    cache_key = _build_cache_key(df, config)

    if (
        "detection_results" in st.session_state
        and st.session_state.get("detection_cache_key") == cache_key
    ):
        return (
            st.session_state["detection_results"],
            st.session_state.get("detection_y_true"),
        )

    # Prepare data
    value_col = _get_value_col(df)
    series = df[value_col].dropna()

    # Extract ground truth if available
    y_true: Optional[np.ndarray] = None
    if "is_anomaly" in df.columns:
        raw_labels = df["is_anomaly"].values
        # Convert 0/1 to -1/1  (1 = anomaly -> -1, 0 = normal -> 1)
        y_true = np.where(raw_labels == 1, -1, 1).astype(int)

    X = series.values.reshape(-1, 1)

    from src.detector import AnomalyShield  # noqa: PLC0415
    from src.models import (  # noqa: PLC0415
        AutoencoderDetector,
        EllipticEnvelopeDetector,
        IsolationForestDetector,
        LOFDetector,
    )

    shield = AnomalyShield()
    contamination = config["contamination"]

    model_map = {
        "Isolation Forest": lambda: IsolationForestDetector(
            name="Isolation Forest",
            contamination=contamination,
        ),
        "LOF": lambda: LOFDetector(
            name="LOF",
            contamination=contamination,
        ),
        "Elliptic Envelope": lambda: EllipticEnvelopeDetector(
            name="Elliptic Envelope",
            contamination=contamination,
        ),
        "Autoencoder": lambda: AutoencoderDetector(
            name="Autoencoder",
            epochs=config["ae_epochs"],
        ),
    }

    selected = config.get("models", [])
    if not selected:
        st.warning("No detectors selected. Please choose at least one.")
        return {}, y_true

    for model_name in selected:
        if model_name in model_map:
            shield.add_detector(model_map[model_name]())

    with st.spinner("Training detectors and running detection..."):
        results = shield.run_all(X, y_true=y_true)

    # Cache
    st.session_state["detection_results"] = results
    st.session_state["detection_y_true"] = y_true
    st.session_state["detection_cache_key"] = cache_key

    return results, y_true


def _build_cache_key(df: pd.DataFrame, config: dict) -> str:
    """Produce a lightweight string key for caching detection results."""
    return (
        f"{len(df)}"
        f"_{df.index.min()}_{df.index.max()}"
        f"_{sorted(config.get('models', []))}"
        f"_{config['contamination']}"
        f"_{config['ae_epochs']}"
    )


# ---------------------------------------------------------------------------
# Plot helpers used by streamlit_app.py
# ---------------------------------------------------------------------------

def plot_anomalies_overlay(
    df: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    detector_name: str,
) -> go.Figure:
    """Time series with anomalies overlaid as red markers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time series DataFrame.
    predictions : np.ndarray
        -1/1 prediction array aligned with *df*.
    scores : np.ndarray
        Anomaly scores aligned with *df* (higher = more anomalous).
    detector_name : str
        Used in the chart title and legend.
    """
    value_col = _get_value_col(df)

    # Predictions may be shorter than df due to window sliding (Autoencoder)
    n = len(predictions)
    plot_df = df.iloc[-n:] if len(df) > n else df

    anomaly_mask = predictions == -1
    normal_idx = plot_df.index[~anomaly_mask]
    anomaly_idx = plot_df.index[anomaly_mask]

    fig = go.Figure()

    # Normal points
    fig.add_trace(
        go.Scatter(
            x=normal_idx,
            y=plot_df.loc[normal_idx, value_col] if len(normal_idx) > 0 else [],
            mode="lines",
            name="Normal",
            line={"color": "#4C9BE8", "width": 1.5},
        )
    )

    # Anomaly markers
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=anomaly_idx,
                y=plot_df.loc[anomaly_idx, value_col],
                mode="markers",
                name="Anomaly",
                marker={"color": "#FF4B4B", "size": 9, "symbol": "x-open-dot"},
            )
        )

    fig.update_layout(
        title=f"{detector_name} — Detected Anomalies",
        xaxis_title="Date",
        yaxis_title=value_col,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_anomaly_scores(
    df: pd.DataFrame,
    scores: np.ndarray,
    detector_name: str,
) -> go.Figure:
    """Bar/area chart of anomaly scores over time.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time series DataFrame.
    scores : np.ndarray
        Anomaly scores aligned with the tail of *df*.
    detector_name : str
        Used in the chart title.
    """
    n = len(scores)
    dates = df.index[-n:] if len(df) > n else df.index

    # Normalise to [0, 1] for visual clarity
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-10)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=norm,
            name="Score (normalised)",
            marker={
                "color": norm,
                "colorscale": "RdYlGn_r",
                "showscale": False,
            },
        )
    )

    fig.update_layout(
        title=f"{detector_name} — Anomaly Scores (normalised)",
        xaxis_title="Date",
        yaxis_title="Score",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        height=280,
        showlegend=False,
    )
    return fig


def plot_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all metrics across detectors.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame returned by ``AnomalyShield.compare()``.
    """
    metrics = [c for c in comparison_df.columns if c in ("accuracy", "precision", "recall", "f1", "auc_roc")]
    if not metrics:
        metrics = comparison_df.columns.tolist()

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=metric.upper(),
                x=comparison_df.index.tolist(),
                y=comparison_df[metric].round(4).tolist(),
                marker_color=palette[i % len(palette)],
                text=comparison_df[metric].round(3).astype(str),
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Detector Comparison — All Metrics",
        xaxis_title="Detector",
        yaxis_title="Score",
        yaxis={"range": [0, 1.15]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        template="plotly_white",
        height=420,
    )
    return fig


def plot_roc_curves(
    results: dict,
    y_true: np.ndarray,
) -> go.Figure:
    """ROC curves for all detectors that have scores.

    Parameters
    ----------
    results : dict
        Mapping of detector name to result dict (with ``scores`` key).
    y_true : np.ndarray
        Ground truth labels (-1/1 convention, -1 = anomaly).
    """
    from sklearn.metrics import roc_curve, auc  # noqa: PLC0415

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    y_bin = (y_true == -1).astype(int)

    for i, (name, result) in enumerate(results.items()):
        scores = result.get("scores")
        if scores is None:
            continue

        n = len(scores)
        y_b = y_bin[-n:] if len(y_bin) > n else y_bin

        try:
            fpr, tpr, _ = roc_curve(y_b, scores)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={roc_auc:.3f})",
                    line={"color": palette[i % len(palette)], "width": 2},
                )
            )
        except ValueError:
            continue

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": "grey", "width": 1},
        )
    )

    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        margin={"l": 40, "r": 20, "t": 50, "b": 80},
        template="plotly_white",
        height=420,
    )
    return fig


def plot_prophet_forecast(forecast_df: pd.DataFrame, history_df: pd.DataFrame) -> go.Figure:
    """Plot Prophet forecast with confidence band and historical data.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output from ``ProphetForecaster.predict()`` with columns
        ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    history_df : pd.DataFrame
        Original time series DataFrame (DatetimeIndex + value column).
    """
    value_col = _get_value_col(history_df)

    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]),
            y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(76, 155, 232, 0.15)",
            line={"color": "rgba(0,0,0,0)"},
            name="95% interval",
            showlegend=True,
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat"],
            mode="lines",
            name="Forecast (yhat)",
            line={"color": "#4C9BE8", "width": 2, "dash": "dot"},
        )
    )

    # Historical values
    fig.add_trace(
        go.Scatter(
            x=history_df.index,
            y=history_df[value_col],
            mode="lines",
            name="Actual",
            line={"color": "#2D3748", "width": 1.5},
        )
    )

    fig.update_layout(
        title="Prophet Forecast",
        xaxis_title="Date",
        yaxis_title=value_col,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_prophet_anomalies(anomaly_df: pd.DataFrame) -> go.Figure:
    """Plot actual values vs Prophet bounds with anomaly highlights.

    Parameters
    ----------
    anomaly_df : pd.DataFrame
        Output from ``ProphetForecaster.detect_anomalies()`` — a copy of the
        input DataFrame enriched with ``yhat``, ``yhat_lower``,
        ``yhat_upper``, and ``is_anomaly`` columns.
    """
    value_col = "value"
    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=list(anomaly_df.index) + list(anomaly_df.index[::-1]),
            y=list(anomaly_df["yhat_upper"]) + list(anomaly_df["yhat_lower"].iloc[::-1]),
            fill="toself",
            fillcolor="rgba(76, 155, 232, 0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name="Prediction interval",
        )
    )

    # Actual values
    normal = anomaly_df[~anomaly_df["is_anomaly"]]
    anomalies = anomaly_df[anomaly_df["is_anomaly"]]

    fig.add_trace(
        go.Scatter(
            x=normal.index,
            y=normal[value_col],
            mode="lines",
            name="Normal",
            line={"color": "#4C9BE8", "width": 1.5},
        )
    )

    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies[value_col],
                mode="markers",
                name="Anomaly (Prophet)",
                marker={"color": "#FF4B4B", "size": 9, "symbol": "x-open-dot"},
            )
        )

    fig.update_layout(
        title="Prophet — Detected Anomalies",
        xaxis_title="Date",
        yaxis_title=value_col,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    return fig
