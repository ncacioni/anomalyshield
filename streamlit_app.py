"""AnomalyShield — Streamlit dashboard entry point."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date

# ---------------------------------------------------------------------------
# Path bootstrap — must come before any src imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from src.visualization.dashboard import (
    load_data,
    plot_anomalies_overlay,
    plot_anomaly_scores,
    plot_comparison_bar,
    plot_prophet_anomalies,
    plot_prophet_forecast,
    plot_roc_curves,
    run_detection,
    show_data_overview,
    sidebar_config,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AnomalyShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS — avoids hardcoded colors by using Streamlit's own tokens
# via CSS variables that Streamlit itself injects.
st.markdown(
    """
    <style>
    /* Give metric cards a slight card feel without overriding theme colors */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    /* Tighten tab strip spacing */
    div[data-testid="stTabs"] button {
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
config = sidebar_config()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ AnomalyShield")
st.markdown(
    "Real-time anomaly detection for time series data using "
    "Isolation Forest, LOF, Elliptic Envelope, LSTM Autoencoder, and Prophet."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df: pd.DataFrame | None = load_data(config)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
TAB_DATA, TAB_DETECT, TAB_COMPARE, TAB_FORECAST, TAB_REPORT = st.tabs(
    [
        "Data Explorer",
        "Anomaly Detection",
        "Comparison",
        "Forecasting",
        "Report",
    ]
)

# ===========================================================================
# TAB 1 — Data Explorer
# ===========================================================================
with TAB_DATA:
    if df is None:
        st.info("Configure a data source in the sidebar to get started.")
    else:
        show_data_overview(df)

# ===========================================================================
# TAB 2 — Anomaly Detection
# ===========================================================================
with TAB_DETECT:
    if df is None:
        st.info("Load data first (Data Explorer tab).")
    elif not config["models"]:
        st.warning("Select at least one detector in the sidebar.")
    else:
        run_btn = st.button(
            "Run Detection",
            type="primary",
            use_container_width=False,
            help="Train selected detectors and flag anomalies.",
        )

        # Run (or use cached) results when button pressed or cache exists
        has_cache = "detection_results" in st.session_state
        if run_btn or has_cache:
            if run_btn:
                # Clear cache so the run is fresh
                st.session_state.pop("detection_results", None)
                st.session_state.pop("detection_cache_key", None)

            results, y_true = run_detection(df, config)

            if results:
                # Summary row
                cols = st.columns(len(results))
                for col, (name, result) in zip(cols, results.items()):
                    preds = result["predictions"]
                    n_anomalies = int((preds == -1).sum())
                    pct = n_anomalies / len(preds) * 100
                    col.metric(name, f"{n_anomalies} anomalies", f"{pct:.1f}% of data")

                st.divider()

                # Per-detector charts
                for det_name, result in results.items():
                    preds = result["predictions"]
                    scores = result["scores"]

                    with st.expander(f"{det_name}", expanded=True):
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            fig_overlay = plot_anomalies_overlay(df, preds, scores, det_name)
                            st.plotly_chart(fig_overlay, use_container_width=True)
                        with col_b:
                            fig_scores = plot_anomaly_scores(df, scores, det_name)
                            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("Press **Run Detection** to train the selected models.")

# ===========================================================================
# TAB 3 — Comparison
# ===========================================================================
with TAB_COMPARE:
    if df is None:
        st.info("Load data first (Data Explorer tab).")
    elif "detection_results" not in st.session_state:
        st.info("Run detection first (Anomaly Detection tab).")
    else:
        results = st.session_state["detection_results"]
        y_true = st.session_state.get("detection_y_true")

        if y_true is None:
            st.warning(
                "Ground truth labels are not available for this dataset. "
                "The Comparison tab requires an ``is_anomaly`` column in the data. "
                "Use the **Sample Data** source or upload a CSV with that column."
            )
        else:
            # Metrics table
            try:
                from src.detector import AnomalyShield  # noqa: PLC0415

                shield_for_compare = AnomalyShield()
                shield_for_compare.results = results  # inject cached results

                comparison_df = shield_for_compare.compare()

                st.subheader("Metrics Table")
                display_df = comparison_df.copy()
                for col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
                st.dataframe(display_df, use_container_width=True)

                st.divider()

                # Comparison bar chart
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.subheader("Metric Comparison")
                    fig_bar = plot_comparison_bar(comparison_df)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_right:
                    st.subheader("ROC Curves")
                    fig_roc = plot_roc_curves(results, y_true)
                    st.plotly_chart(fig_roc, use_container_width=True)

            except ValueError:
                st.error("Could not compute metrics. Ensure ground truth labels are present.")

# ===========================================================================
# TAB 4 — Forecasting (Prophet)
# ===========================================================================
with TAB_FORECAST:
    if df is None:
        st.info("Load data first (Data Explorer tab).")
    elif not config["use_prophet"]:
        st.info("Enable **Prophet forecasting** in the sidebar to use this tab.")
    else:
        st.subheader("Prophet Forecasting")

        # Ensure data has a 'value' column that Prophet needs
        value_col_for_prophet = None
        for candidate in ("value", "Close", "close"):
            if candidate in df.columns:
                value_col_for_prophet = candidate
                break
        if value_col_for_prophet is None:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            non_label = [c for c in num_cols if c != "is_anomaly"]
            value_col_for_prophet = non_label[0] if non_label else None

        if value_col_for_prophet is None:
            st.error("No numeric value column found for Prophet.")
        else:
            # Prepare a Prophet-compatible df with a 'value' column
            prophet_input = df[[value_col_for_prophet]].copy()
            if value_col_for_prophet != "value":
                prophet_input = prophet_input.rename(columns={value_col_for_prophet: "value"})

            forecast_periods = st.slider(
                "Forecast horizon (days)",
                min_value=7,
                max_value=365,
                value=30,
                step=7,
                help="Number of future days to forecast beyond the last data point.",
            )

            col_run_p, _ = st.columns([1, 4])
            run_prophet_btn = col_run_p.button(
                "Run Prophet",
                type="primary",
                help="Fit Prophet and generate forecast + anomaly detection.",
            )

            prophet_cache_key = f"prophet_{len(df)}_{forecast_periods}_{config['data_source']}"

            if run_prophet_btn or st.session_state.get("prophet_cache_key") == prophet_cache_key:
                if run_prophet_btn:
                    st.session_state.pop("prophet_forecast", None)
                    st.session_state.pop("prophet_anomalies", None)

                if "prophet_forecast" not in st.session_state:
                    try:
                        from src.models import ProphetForecaster  # noqa: PLC0415

                        with st.spinner("Fitting Prophet model (this may take a moment)..."):
                            forecaster = ProphetForecaster()
                            anomaly_df = forecaster.detect_anomalies(prophet_input)
                            forecast_df = forecaster.predict(periods=forecast_periods)

                        st.session_state["prophet_forecast"] = forecast_df
                        st.session_state["prophet_anomalies"] = anomaly_df
                        st.session_state["prophet_cache_key"] = prophet_cache_key

                    except Exception:  # noqa: BLE001
                        st.error("Prophet model failed. Check your data format and try again.")

                if "prophet_forecast" in st.session_state:
                    forecast_df = st.session_state["prophet_forecast"]
                    anomaly_df = st.session_state["prophet_anomalies"]

                    # Forecast plot
                    st.subheader("Forecast")
                    fig_fc = plot_prophet_forecast(forecast_df, prophet_input)
                    st.plotly_chart(fig_fc, use_container_width=True)

                    # Anomaly detection plot
                    st.subheader("Anomalies Detected by Prophet")
                    n_prophet_anomalies = int(anomaly_df["is_anomaly"].sum())
                    pct_p = n_prophet_anomalies / len(anomaly_df) * 100
                    st.metric(
                        "Points outside prediction interval",
                        f"{n_prophet_anomalies} ({pct_p:.1f}%)",
                    )

                    fig_pa = plot_prophet_anomalies(anomaly_df)
                    st.plotly_chart(fig_pa, use_container_width=True)

                    with st.expander("Forecast data"):
                        st.dataframe(
                            forecast_df.rename(columns={"ds": "date"}).set_index("date"),
                            use_container_width=True,
                        )
            else:
                st.info("Press **Run Prophet** to fit the model and generate a forecast.")

# ===========================================================================
# TAB 5 — Report
# ===========================================================================
with TAB_REPORT:
    if df is None:
        st.info("Load data first (Data Explorer tab).")
    elif "detection_results" not in st.session_state:
        st.info("Run detection first (Anomaly Detection tab).")
    else:
        results = st.session_state["detection_results"]

        col_gen, _ = st.columns([1, 4])
        generate_btn = col_gen.button(
            "Generate Report",
            type="primary",
            help="Produce a Markdown summary of all detection results.",
        )

        if generate_btn or "report_markdown" in st.session_state:
            if generate_btn:
                try:
                    from src.utils import generate_report  # noqa: PLC0415

                    with st.spinner("Generating report..."):
                        tmp_path = os.path.join(
                            tempfile.gettempdir(),
                            "anomalyshield_report.md",
                        )
                        report_md = generate_report(results, output_path=tmp_path)
                        st.session_state["report_markdown"] = report_md
                        st.session_state["report_path"] = tmp_path

                except Exception:  # noqa: BLE001
                    st.error("Failed to generate report. Please try again.")

            if "report_markdown" in st.session_state:
                report_md = st.session_state["report_markdown"]

                st.markdown(report_md)

                st.divider()

                st.download_button(
                    label="Download Report (.md)",
                    data=report_md.encode("utf-8"),
                    file_name="anomalyshield_report.md",
                    mime="text/markdown",
                    help="Save the Markdown report to your computer.",
                )
        else:
            st.info("Press **Generate Report** to create a full Markdown summary.")
