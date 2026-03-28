"""
Creates the SQLite database and schema.
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "data/finance.db"

def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id  INTEGER PRIMARY KEY,
            date            TEXT NOT NULL,
            merchant        TEXT NOT NULL,
            category        TEXT NOT NULL,
            amount          REAL NOT NULL,
            type            TEXT CHECK(type IN ('debit','credit')) NOT NULL,
            is_anomaly      INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS monthly_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            year_month      TEXT NOT NULL,
            category        TEXT NOT NULL,
            total_spent     REAL,
            avg_transaction REAL,
            tx_count        INTEGER,
            UNIQUE(year_month, category)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            year_month      TEXT NOT NULL,
            category        TEXT NOT NULL,
            predicted_spend REAL,
            created_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(year_month, category)
        );
    """)
    conn.commit()
    print("✅ Schema created.")


def load_csv_to_db(conn: sqlite3.Connection, csv_path: str):
    df = pd.read_csv(csv_path, index_col="transaction_id")
    df["is_anomaly"] = df["is_anomaly"].astype(int)
    df.to_sql("transactions", conn, if_exists="replace", index=True)
    print(f"✅ Loaded {len(df)} rows into transactions table.")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    csv_path = "data/raw/transactions.csv"
    if Path(csv_path).exists():
        load_csv_to_db(conn, csv_path)
    else:
        print("⚠️  No CSV found. Run data/generate_data.py first.")

    conn.close()
