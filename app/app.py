
"""
app.py  –  BudgetPulse
Run with:  streamlit run app/app.py
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "data/finance.db"

st.set_page_config(
    page_title="BudgetPulse",

    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .anomaly-badge {
        background: #ff4b4b22;
        border: 1px solid #ff4b4b;
        border-radius: 6px;
        padding: 2px 8px;
        color: #ff4b4b;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_transactions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    conn.close()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM monthly_summary", conn)
    conn.close()
    return df


@st.cache_data
def load_predictions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def detect_anomalies_live(df: pd.DataFrame) -> pd.DataFrame:
    """Fast Z-score anomaly detection on the fly."""
    expenses = df[df["type"] == "debit"].copy()
    expenses["z_score"] = expenses.groupby("category")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return expenses[expenses["z_score"] > 2.5].sort_values("amount", ascending=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BudgetPulse")
    st.caption("BudgetPulse")
    st.divider()

    df_all = load_transactions()
    months = sorted(df_all["year_month"].unique())

    selected_months = st.multiselect(
        "Filter by Month",
        options=months,
        default=months[-3:],
    )

    categories = sorted(df_all["category"].unique())
    selected_cats = st.multiselect(
        "Filter by Category",
        options=categories,
        default=categories,
    )

    st.divider()
    uploaded = st.file_uploader("📂 Upload your own CSV", type=["csv"])
    if uploaded:
        df_all = pd.read_csv(uploaded, parse_dates=["date"])
        df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)
        st.success(f"Loaded {len(df_all)} rows!")

# ── Filter Data ───────────────────────────────────────────────────────────────
df = df_all[
    (df_all["year_month"].isin(selected_months)) &
    (df_all["category"].isin(selected_cats))
]
expenses = df[df["type"] == "debit"]
income   = df[df["type"] == "credit"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("BudgetPulse")
st.caption(f"Showing {len(df):,} transactions across {len(selected_months)} months")

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total_income  = income["amount"].sum()
total_spent   = expenses["amount"].sum()
net           = total_income - total_spent
avg_per_month = total_spent / max(len(selected_months), 1)

k1.metric("Total Income",    f"${total_income:,.0f}")
k2.metric("Total Spent",     f"${total_spent:,.0f}")
k3.metric("Net Savings",     f"${net:,.0f}",     delta=f"${net/max(total_income,1)*100:.1f}% saved")
k4.metric("Avg / Month",     f"${avg_per_month:,.0f}")

st.divider()

# ── Row 1: Spending Over Time + Category Breakdown ────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Monthly Spending Trend")
    monthly = expenses.groupby("year_month")["amount"].sum().reset_index()
    monthly.columns = ["Month", "Spent"]
    fig = px.bar(
        monthly, x="Month", y="Spent",
        color="Spent",
        color_continuous_scale="Blues",
        labels={"Spent": "Amount ($)"},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Spending by Category")
    by_cat = expenses.groupby("category")["amount"].sum().reset_index()
    fig2 = px.pie(
        by_cat, values="amount", names="category",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig2.update_layout(showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Forecast + Anomalies ───────────────────────────────────────────────
col3, col4 = st.columns([2, 3])

with col3:
    st.subheader("Next Month Forecast")
    preds = load_predictions()
    if preds.empty:
        st.info("Run `src/models/forecaster.py` to generate predictions.")
    else:
        preds_display = preds[["category", "predicted_spend"]].copy()
        preds_display.columns = ["Category", "Predicted ($)"]
        preds_display = preds_display.sort_values("Predicted ($)", ascending=False)

        fig3 = px.bar(
            preds_display, x="Predicted ($)", y="Category",
            orientation="h",
            color="Predicted ($)",
            color_continuous_scale="Teal",
        )
        fig3.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Anomalous Transactions")
    anomalies = detect_anomalies_live(df)
    if anomalies.empty:
        st.success("No anomalies detected in selected period.")
    else:
        st.warning(f"{len(anomalies)} unusual transactions flagged")
        display_cols = ["date", "merchant", "category", "amount", "z_score"]
        anomalies_show = anomalies[display_cols].copy()
        anomalies_show["amount"] = anomalies_show["amount"].apply(lambda x: f"${x:,.2f}")
        anomalies_show["z_score"] = anomalies_show["z_score"].apply(lambda x: f"{x:.1f}σ")
        anomalies_show["date"] = anomalies_show["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(anomalies_show, use_container_width=True, hide_index=True)

# ── Row 3: Category Deep Dive ─────────────────────────────────────────────────
st.divider()
st.subheader("Category Deep Dive")

cat_select = st.selectbox("Select a category", options=sorted(expenses["category"].unique()))
cat_df = expenses[expenses["category"] == cat_select]

c1, c2, c3 = st.columns(3)
c1.metric("Total Spent",       f"${cat_df['amount'].sum():,.2f}")
c2.metric("Avg Transaction",   f"${cat_df['amount'].mean():,.2f}")
c3.metric("# Transactions",    f"{len(cat_df)}")

fig4 = px.scatter(
    cat_df, x="date", y="amount",
    color="merchant",
    size="amount",
    hover_data=["merchant", "amount"],
    title=f"{cat_select} — All Transactions",
)
fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Python · Pandas · scikit-learn · Streamlit · Plotly · SQLite")
