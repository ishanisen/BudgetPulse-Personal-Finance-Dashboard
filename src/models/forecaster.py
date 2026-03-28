"""
Predicts next month's spending per category using Linear Regression.

"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

DB_PATH = "data/finance.db"


def load_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM monthly_summary", conn)
    df["period_index"] = pd.to_datetime(df["year_month"]).dt.to_period("M")
    df["period_int"] = df["period_index"].apply(lambda p: p.ordinal)
    return df


def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each category, fit a linear regression on period → total_spent
    and predict one month ahead.
    """
    categories = df["category"].unique()
    next_period = df["period_int"].max() + 1
    next_month = pd.Period(ordinal=next_period, freq="M").strftime("%Y-%m")

    predictions = []

    for cat in categories:
        cat_df = df[df["category"] == cat].sort_values("period_int")

        if len(cat_df) < 3:
            continue  # not enough data

        X = cat_df[["period_int"]].values
        y = cat_df["total_spent"].values

        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict([[next_period]])[0]
        pred = max(0, round(pred, 2))  # no negative spending

        # MAE on training data as confidence proxy
        y_hat = model.predict(X)
        mae = round(mean_absolute_error(y, y_hat), 2)

        predictions.append({
            "year_month": next_month,
            "category": cat,
            "predicted_spend": pred,
            "mae": mae,
        })

        print(f"  {cat:15s} → ${pred:8.2f}  (MAE ${mae:.2f})")

    return pd.DataFrame(predictions)


def save_predictions(conn: sqlite3.Connection, preds: pd.DataFrame):
    preds[["year_month", "category", "predicted_spend"]].to_sql(
        "predictions", conn, if_exists="replace", index=False
    )
    print(f"✅ Saved {len(preds)} predictions.")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = load_summary(conn)
    print(f"Training on {df['year_month'].nunique()} months of data...\n")
    preds = train_and_predict(df)
    save_predictions(conn, preds)
    conn.close()
