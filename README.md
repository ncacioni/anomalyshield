<p align="center">
  <img src="assets/anomalyshield-logo.png" alt="AnomalyShield Logo" width="250">
</p>

<h1 align="center">AnomalyShield</h1>

<p align="center">
  Time series anomaly detection platform — multiple methods, interactive visualization, and forecasting.
</p>

<p align="center">
  <a href="https://github.com/ncacioni/anomalyshield/releases"><img src="https://img.shields.io/github/v/release/ncacioni/anomalyshield?style=flat-square&color=blue" alt="Release"></a>
  <a href="https://github.com/ncacioni/anomalyshield/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ncacioni/anomalyshield?style=flat-square" alt="License"></a>
  <a href="https://github.com/ncacioni/anomalyshield/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ncacioni/anomalyshield/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/ncacioni/anomalyshield/actions/workflows/codeql.yml"><img src="https://img.shields.io/github/actions/workflow/status/ncacioni/anomalyshield/codeql.yml?style=flat-square&label=CodeQL" alt="CodeQL"></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-221%20passing-brightgreen?style=flat-square" alt="Tests">
</p>

---

## Features

- **Multiple detection methods**: Isolation Forest, Local Outlier Factor, Elliptic Envelope, LSTM Autoencoder
- **Time series forecasting**: Facebook Prophet with anomaly detection via prediction intervals
- **Interactive visualization**: 7 Plotly chart types (time series, scores, comparison, confusion matrix, ROC curves, forecast)
- **Streamlit dashboard**: 5-tab interface (data explorer, detection, comparison, forecasting, reports)
- **Flexible data sources**: CSV files, Yahoo Finance API, PostgreSQL databases
- **Evaluation framework**: Accuracy, precision, recall, F1, AUC-ROC with ensemble predictions and Markdown reports

## Quick Start

```bash
# Clone and install
git clone https://github.com/ncacioni/anomalyshield.git
cd anomalyshield
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run streamlit_app.py
```

## Project Structure

```
src/
├── detector.py              # Core AnomalyShield orchestrator + BaseDetector
├── models/
│   ├── isolation_forest.py  # Isolation Forest detector
│   ├── lof.py               # Local Outlier Factor detector
│   ├── elliptic_envelope.py # Elliptic Envelope detector
│   ├── autoencoder.py       # LSTM Autoencoder detector
│   └── prophet_model.py     # Prophet forecaster
├── data/
│   ├── loader.py            # CSV/DataFrame ingestion
│   ├── preprocessor.py      # Normalization, features, missing values
│   └── sources.py           # Yahoo Finance + PostgreSQL adapters
├── visualization/
│   ├── plots.py             # 7 Plotly chart functions
│   └── dashboard.py         # Streamlit components
└── utils.py                 # Evaluation metrics + report generation

notebooks/
├── 01_exploration.ipynb     # Exploratory Data Analysis
├── 02_detection_methods.ipynb # Detection model walkthrough
└── 03_evaluation.ipynb      # Model comparison

tests/                       # 221 tests (data, detector, models)
streamlit_app.py             # Dashboard entry point
```

## Usage

```python
from src.detector import AnomalyShield
from src.models import IsolationForestDetector, LOFDetector
from src.data.loader import TimeSeriesLoader

# Load data
df = TimeSeriesLoader.from_csv("data.csv", date_col="date", value_col="value")

# Detect anomalies
shield = AnomalyShield()
shield.add_detector("iforest", IsolationForestDetector(contamination=0.05))
shield.add_detector("lof", LOFDetector(n_neighbors=20))

results = shield.run_all(df[["value"]].values)
ensemble = shield.ensemble_predictions(method="majority")
```

## Detection Methods

| Method | Type | Best For |
|--------|------|----------|
| Isolation Forest | Tree-based | General-purpose, high-dimensional data |
| Local Outlier Factor | Density-based | Clusters with varying density |
| Elliptic Envelope | Statistical | Gaussian-distributed data |
| LSTM Autoencoder | Deep learning | Temporal patterns, sequences |
| Prophet | Forecasting | Seasonal time series with trend |

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn, PyOD, PyTorch |
| Forecasting | Prophet |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Database | PostgreSQL (SQLAlchemy) |
| APIs | yfinance |

## License

[MIT](LICENSE)
