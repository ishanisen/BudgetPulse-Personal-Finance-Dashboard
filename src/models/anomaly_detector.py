"""
anomaly_detector.py
Flags unusual transactions using Z-score and Isolation Forest.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

DB_PATH = "data/finance.db"


def load_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM transactions WHERE type='debit'", conn)
    return df


def zscore_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """Flag transactions where amount is > threshold std deviations above category mean."""
    df = df.copy()
    df["z_score"] = df.groupby("category")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df["zscore_flag"] = df["z_score"] > threshold
    return df


def isolation_forest_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Use Isolation Forest for multivariate anomaly detection."""
    df = df.copy()

    # Encode category as numeric
    df["category_enc"] = df["category"].astype("category").cat.codes

    features = df[["amount", "category_enc"]].values
    clf = IsolationForest(contamination=0.03, random_state=42)
    df["iforest_flag"] = clf.fit_predict(features) == -1

    return df


def detect_all(conn: sqlite3.Connection) -> pd.DataFrame:
    df = load_transactions(conn)
    df = zscore_anomalies(df)
    df = isolation_forest_anomalies(df)

    # Combine: flag if either method agrees
    df["anomaly_detected"] = df["zscore_flag"] | df["iforest_flag"]

    flagged = df[df["anomaly_detected"]][
        ["transaction_id", "date", "merchant", "category", "amount", "z_score"]
    ].sort_values("amount", ascending=False)

    print(f"Detected {len(flagged)} anomalous transactions out of {len(df)}")
    return flagged


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    flagged = detect_all(conn)
    print(flagged.head(10).to_string(index=False))
    conn.close()
