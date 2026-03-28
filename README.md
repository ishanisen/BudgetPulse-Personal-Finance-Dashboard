# BudgetPulse

> An end-to-end data science project - data pipeline, SQL storage, ML forecasting, anomaly detection, and interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)
![Plotly](https://img.shields.io/badge/Plotly-5.18-blueviolet?logo=plotly)

---

## Features

| Feature | Description |
|---|---|
|**Spending Dashboard** | Monthly trends, category breakdowns, KPI cards |
|**ML Forecasting** | Predicts next month's spending per category (Linear Regression) |
|**Anomaly Detection** | Flags unusual transactions using Z-score + Isolation Forest |
|**SQL Backend** | Full SQLite schema — transactions, summaries, predictions |
|**CSV Upload** | Plug in your own exported bank data |
|**Category Deep Dive** | Drill into any spending category |
---

## Architecture

```
budgetpulse/
├── data/
│   ├── generate_data.py      # Synthetic data generator
│   └── raw/                  # Raw CSVs live here
├── src/
│   ├── pipeline/
│   │   ├── db_setup.py       # SQLite schema + CSV loader
│   │   └── clean.py          # Cleaning & monthly summary pipeline
│   └── models/
│       ├── forecaster.py     # Linear Regression spending forecaster
│       └── anomaly_detector.py # Z-score + Isolation Forest
├── app/
│   └── app.py                # Streamlit dashboard
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/budgetpulse.git
cd budgetpulse
pip install -r requirements.txt

# 2. Generate sample data
python data/generate_data.py

# 3. Set up database & run pipeline
python src/pipeline/db_setup.py
python src/pipeline/clean.py

# 4. Train models
python src/models/forecaster.py
python src/models/anomaly_detector.py

# 5. Launch dashboard
streamlit run app/app.py
```

---

## ML Models

### Spending Forecaster
- **Algorithm**: Linear Regression (per category)
- **Features**: Month ordinal index
- **Output**: Predicted spend for each category next month
- **Metric**: Mean Absolute Error (MAE) reported per category

### Anomaly Detector
- **Algorithm 1**: Z-score (flags transactions > 2.5σ above category mean)
- **Algorithm 2**: Isolation Forest (multivariate, contamination=3%)
- **Ensemble**: Transaction flagged if either method agrees

---

## Sample Insights

- Groceries and Dining account for the largest monthly discretionary spend
- Utility bills show low variance - ideal for budget planning
- ~2-3% of transactions are statistical outliers

---

## Using Your Own Data

Export transactions from your bank as CSV with these columns:

```
date, merchant, category, amount, type
```

Then upload the CSV via the sidebar in the dashboard.

---

## Roadmap
- [ ] Prophet time series forecasting
- [ ] Budget goal setting & alerts
- [ ] Multi-account support
- [ ] PostgreSQL migration for production
- [ ] React frontend version
---

## Author

**Ishani Sen** · Computer Engineering Student  

