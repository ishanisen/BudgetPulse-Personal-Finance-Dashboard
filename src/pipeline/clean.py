"""
Cleans raw transaction data and computes monthly summaries into the DB.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "data/finance.db"


def load_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()

    # Enforce positive amounts
    df = df[df["amount"] > 0].copy()

    # Add time features
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # Separate income vs expenses
    df["net_amount"] = df.apply(
        lambda r: r["amount"] if r["type"] == "credit" else -r["amount"], axis=1
    )

    return df


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["type"] == "debit"]
    summary = (
        expenses.groupby(["year_month", "category"])
        .agg(
            total_spent=("amount", "sum"),
            avg_transaction=("amount", "mean"),
            tx_count=("amount", "count"),
        )
        .reset_index()
    )
    summary["total_spent"] = summary["total_spent"].round(2)
    summary["avg_transaction"] = summary["avg_transaction"].round(2)
    return summary


def save_summary(conn: sqlite3.Connection, summary: pd.DataFrame):
    summary.to_sql("monthly_summary", conn, if_exists="replace", index=False)
    print(f"✅ Saved {len(summary)} monthly summary rows.")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = load_transactions(conn)
    df = clean(df)
    summary = compute_monthly_summary(df)
    save_summary(conn, summary)
    conn.close()
    print("✅ Pipeline complete.")
