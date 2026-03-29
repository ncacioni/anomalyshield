"""Plotly-based visualization functions for AnomalyShield."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve

# ---------------------------------------------------------------------------
# Design tokens — all visual values defined once here
# ---------------------------------------------------------------------------

_COLORS = {
    "normal": "#4C9BE8",        # calm blue for normal data points / lines
    "anomaly": "#E84C4C",       # vivid red for anomaly markers
    "threshold": "#E84C4C",     # same red for threshold lines
    "forecast": "#6C63FF",      # purple for forecast line
    "band_fill": "rgba(108, 99, 255, 0.15)",  # translucent purple for CI band
    "actuals": "#2EC4B6",       # teal for actual values
    "anomaly_point": "#FF6B35", # orange-red for anomaly overlay points
    "grid": "rgba(200, 200, 200, 0.3)",
    "zero_line": "rgba(150, 150, 150, 0.5)",
    "background": "#FAFAFA",
    "paper": "#FFFFFF",
    "text": "#1A1A2E",
    "subtext": "#6B7280",
    "roc_diagonal": "rgba(150, 150, 150, 0.6)",
    # Palette for multi-series charts (detectors, metrics)
    "palette": [
        "#4C9BE8", "#6C63FF", "#2EC4B6", "#F4A261",
        "#E84C4C", "#57CC99", "#C77DFF", "#F77F00",
    ],
}

_LAYOUT_DEFAULTS = dict(
    width=900,
    height=500,
    paper_bgcolor=_COLORS["paper"],
    plot_bgcolor=_COLORS["background"],
    font=dict(family="Inter, system-ui, sans-serif", color=_COLORS["text"], size=13),
    margin=dict(l=70, r=30, t=70, b=60),
    hoverlabel=dict(
        bgcolor=_COLORS["paper"],
        bordercolor=_COLORS["grid"],
        font_size=12,
    ),
)

_AXIS_DEFAULTS = dict(
    gridcolor=_COLORS["grid"],
    zerolinecolor=_COLORS["zero_line"],
    linecolor=_COLORS["grid"],
    showgrid=True,
    zeroline=False,
    tickfont=dict(size=11, color=_COLORS["subtext"]),
)


def _apply_axis_defaults(fig: go.Figure) -> None:
    """Apply consistent axis styling to all axes in a figure."""
    fig.update_xaxes(**_AXIS_DEFAULTS)
    fig.update_yaxes(**_AXIS_DEFAULTS)


# ---------------------------------------------------------------------------
# 1. plot_time_series
# ---------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    anomalies: np.ndarray | None = None,
    title: str = "Time Series",
) -> go.Figure:
    """Plot a time series line chart with optional anomaly overlay.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``. The first numeric column (or a
        column named ``"value"``) is used as the y-axis series.
    anomalies : np.ndarray | None
        Integer array of -1/1 labels (length == len(df)).  Points where
        ``anomalies == -1`` are rendered as red scatter markers.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if "value" in df.columns:
        y = df["value"]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("df has no numeric columns to plot.")
        y = df[numeric_cols[0]]

    x = df.index

    fig = go.Figure()

    # --- Main series ---
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Value",
            line=dict(color=_COLORS["normal"], width=1.8),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Value: %{y:.4f}<extra></extra>",
        )
    )

    # --- Anomaly overlay ---
    if anomalies is not None:
        anomalies = np.asarray(anomalies)
        mask = anomalies == -1
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        color=_COLORS["anomaly"],
                        size=8,
                        symbol="circle-open",
                        line=dict(width=2, color=_COLORS["anomaly"]),
                    ),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Anomaly: %{y:.4f}<extra></extra>",
                )
            )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color=_COLORS["text"]), x=0.03),
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    _apply_axis_defaults(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. plot_anomaly_scores
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Anomaly Scores",
) -> go.Figure:
    """Scatter plot of anomaly scores with optional threshold line.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of anomaly scores (higher = more anomalous).
    threshold : float | None
        If provided, a horizontal dashed red line is drawn at this level.
        Points above the threshold are colored differently.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    scores = np.asarray(scores)
    x = np.arange(len(scores))

    fig = go.Figure()

    if threshold is not None:
        # Split into two traces for a clean legend
        normal_mask = scores < threshold
        anomaly_mask = ~normal_mask

        if normal_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[normal_mask],
                    y=scores[normal_mask],
                    mode="markers",
                    name="Normal",
                    marker=dict(color=_COLORS["normal"], size=5, opacity=0.75),
                    hovertemplate="Index: %{x}<br>Score: %{y:.6f}<extra></extra>",
                )
            )

        if anomaly_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[anomaly_mask],
                    y=scores[anomaly_mask],
                    mode="markers",
                    name="Above threshold",
                    marker=dict(color=_COLORS["anomaly"], size=6, opacity=0.9),
                    hovertemplate="Index: %{x}<br>Score: %{y:.6f}<extra></extra>",
                )
            )

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=_COLORS["threshold"],
            line_width=1.5,
            annotation_text=f"Threshold: {threshold:.4f}",
            annotation_position="top right",
            annotation_font=dict(color=_COLORS["threshold"], size=11),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scores,
                mode="markers",
                name="Score",
                marker=dict(color=_COLORS["normal"], size=5, opacity=0.75),
                hovertemplate="Index: %{x}<br>Score: %{y:.6f}<extra></extra>",
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color=_COLORS["text"]), x=0.03),
        xaxis_title="Sample Index",
        yaxis_title="Anomaly Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    _apply_axis_defaults(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. plot_comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "f1",
) -> go.Figure:
    """Grouped bar chart comparing detectors on a single metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``AnomalyShield.compare()``.  Index = detector names,
        columns = metric names.
    metric : str
        Column name to display (e.g. ``"f1"``, ``"accuracy"``).

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if metric not in comparison_df.columns:
        available = list(comparison_df.columns)
        raise ValueError(
            f"Metric '{metric}' not found in comparison_df. "
            f"Available metrics: {available}"
        )

    detector_names = comparison_df.index.tolist()
    values = comparison_df[metric].values

    palette = _COLORS["palette"]
    bar_colors = [palette[i % len(palette)] for i in range(len(detector_names))]

    fig = go.Figure(
        go.Bar(
            x=detector_names,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + metric.upper() + ": %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"Detector Comparison — {metric.upper()}",
            font=dict(size=16, color=_COLORS["text"]),
            x=0.03,
        ),
        xaxis_title="Detector",
        yaxis_title=metric.upper(),
        yaxis=dict(range=[0, min(max(values) * 1.2, 1.05)]),
        showlegend=False,
    )
    _apply_axis_defaults(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. plot_comparison_all_metrics
# ---------------------------------------------------------------------------

def plot_comparison_all_metrics(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing ALL metrics side by side for each detector.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``AnomalyShield.compare()``.  Index = detector names,
        columns = metric names.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    detector_names = comparison_df.index.tolist()
    metrics = comparison_df.columns.tolist()
    palette = _COLORS["palette"]

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        values = comparison_df[metric].values
        fig.add_trace(
            go.Bar(
                name=metric.upper(),
                x=detector_names,
                y=values,
                marker_color=palette[i % len(palette)],
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>" + metric.upper() + ": %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=520,
        title=dict(
            text="Detector Comparison — All Metrics",
            font=dict(size=16, color=_COLORS["text"]),
            x=0.03,
        ),
        xaxis_title="Detector",
        yaxis_title="Score",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=dict(text="Metric", font=dict(size=11)),
        ),
    )
    _apply_axis_defaults(fig)
    return fig


# ---------------------------------------------------------------------------
# 5. plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Annotated heatmap confusion matrix using Plotly.

    Labels follow the sklearn convention: ``-1`` = anomaly, ``1`` = normal.
    Both 0/1 and -1/1 label formats are accepted.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (-1/1 or 0/1).
    y_pred : np.ndarray
        Predicted labels (-1/1).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Normalise to -1/1
    if set(np.unique(y_true)).issubset({0, 1}):
        y_true = np.where(y_true == 0, -1, 1)

    # Build 2x2 matrix manually: rows = actual, cols = predicted
    # Order: [Normal (1), Anomaly (-1)]
    labels_order = [1, -1]   # Normal first, then Anomaly
    label_names = ["Normal", "Anomaly"]

    cm = np.zeros((2, 2), dtype=int)
    for r, actual in enumerate(labels_order):
        for c, predicted in enumerate(labels_order):
            cm[r, c] = int(np.sum((y_true == actual) & (y_pred == predicted)))

    # Annotation text
    annotations = [[str(cm[r, c]) for c in range(2)] for r in range(2)]

    colorscale = [
        [0.0, "#EFF6FF"],
        [0.5, "#93C5FD"],
        [1.0, "#1D4ED8"],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=label_names,
            y=label_names,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=18, color=_COLORS["text"]),
            colorscale=colorscale,
            showscale=True,
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ),
            colorbar=dict(title=dict(text="Count", side="right"), thickness=14),
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        width=600,
        height=500,
        title=dict(text=title, font=dict(size=16, color=_COLORS["text"]), x=0.03),
        xaxis=dict(
            title="Predicted Label",
            tickfont=dict(size=12),
            side="bottom",
        ),
        yaxis=dict(
            title="Actual Label",
            tickfont=dict(size=12),
            autorange="reversed",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. plot_forecast
# ---------------------------------------------------------------------------

def plot_forecast(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame | None = None,
    title: str = "Prophet Forecast",
) -> go.Figure:
    """Plot Prophet forecast with confidence band and optional actuals overlay.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of ``ProphetForecaster.predict()``.  Must have columns:
        ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    actuals_df : pd.DataFrame | None
        Optional DataFrame with a ``DatetimeIndex`` and a ``value`` column.
        If it also contains an ``is_anomaly`` boolean column, those points
        are highlighted as anomalies.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    required = {"ds", "yhat", "yhat_lower", "yhat_upper"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(
            f"forecast_df is missing required columns: {missing}. "
            "Expected output from ProphetForecaster.predict()."
        )

    x_forecast = forecast_df["ds"]
    fig = go.Figure()

    # --- Confidence band (filled area between yhat_lower and yhat_upper) ---
    fig.add_trace(
        go.Scatter(
            x=pd.concat([x_forecast, x_forecast[::-1]]),
            y=pd.concat(
                [forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]
            ),
            fill="toself",
            fillcolor=_COLORS["band_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        )
    )

    # --- Forecast line ---
    fig.add_trace(
        go.Scatter(
            x=x_forecast,
            y=forecast_df["yhat"],
            mode="lines",
            name="Forecast (yhat)",
            line=dict(color=_COLORS["forecast"], width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.4f}<extra></extra>",
        )
    )

    # --- Actuals overlay ---
    if actuals_df is not None:
        if "value" not in actuals_df.columns:
            raise ValueError(
                "actuals_df must contain a 'value' column. "
                "Ensure the DataFrame matches ProphetForecaster input format."
            )

        actual_x = actuals_df.index
        actual_y = actuals_df["value"]

        has_anomaly_col = "is_anomaly" in actuals_df.columns

        if has_anomaly_col:
            normal_mask = ~actuals_df["is_anomaly"].astype(bool)
            anomaly_mask = actuals_df["is_anomaly"].astype(bool)
        else:
            normal_mask = np.ones(len(actuals_df), dtype=bool)
            anomaly_mask = np.zeros(len(actuals_df), dtype=bool)

        # Normal actuals
        if normal_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=actual_x[normal_mask],
                    y=actual_y[normal_mask],
                    mode="markers",
                    name="Actual",
                    marker=dict(
                        color=_COLORS["actuals"],
                        size=4,
                        opacity=0.7,
                    ),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.4f}<extra></extra>"
                    ),
                )
            )

        # Anomalous actuals
        if has_anomaly_col and anomaly_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=actual_x[anomaly_mask],
                    y=actual_y[anomaly_mask],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        color=_COLORS["anomaly"],
                        size=8,
                        symbol="x",
                        line=dict(width=2, color=_COLORS["anomaly"]),
                    ),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>Anomaly: %{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color=_COLORS["text"]), x=0.03),
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    _apply_axis_defaults(fig)
    return fig


# ---------------------------------------------------------------------------
# 7. plot_roc_curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: dict,
    y_true: np.ndarray,
) -> go.Figure:
    """Overlay ROC curves for multiple detectors with AUC in legend labels.

    Parameters
    ----------
    results : dict
        Output of ``AnomalyShield.run_all()``.  Each value is a dict that
        must contain a ``"scores"`` key (np.ndarray of anomaly scores, higher
        = more anomalous).
    y_true : np.ndarray
        Ground truth labels (-1/1 or 0/1).

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    y_true = np.asarray(y_true)

    # Normalise to binary: anomaly=1, normal=0
    if set(np.unique(y_true)).issubset({0, 1}):
        y_true_bin = y_true.astype(int)
    else:
        # -1/1 convention — anomaly is -1
        y_true_bin = (y_true == -1).astype(int)

    palette = _COLORS["palette"]
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        if "scores" not in result:
            continue

        scores = np.asarray(result["scores"])
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            roc_auc = auc(fpr, tpr)
        except ValueError:
            # Single class — skip this detector
            continue

        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {roc_auc:.3f})",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}"
                    f"<br>AUC: {roc_auc:.3f}<extra></extra>"
                ),
            )
        )

    # --- Random classifier diagonal ---
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.500)",
            line=dict(color=_COLORS["roc_diagonal"], width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text="ROC Curves — Detector Comparison",
            font=dict(size=16, color=_COLORS["text"]),
            x=0.03,
        ),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1.02], scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=_COLORS["grid"],
            borderwidth=1,
        ),
        hovermode="closest",
        width=620,
        height=580,
    )
    _apply_axis_defaults(fig)
    return fig
