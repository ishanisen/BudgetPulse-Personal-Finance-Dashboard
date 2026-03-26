"""
generate_data.py
Generates realistic sample transaction data for the finance dashboard.
Run this once to populate data/raw/transactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Groceries":     {"merchants": ["Sobeys", "Loblaws", "Walmart", "Costco"],          "mean": 85,  "std": 30},
    "Dining":        {"merchants": ["Tim Hortons", "McDonald's", "Local Bistro", "Sushi Place"], "mean": 28,  "std": 15},
    "Transport":     {"merchants": ["Esso", "Shell", "Metro Transit", "Uber"],           "mean": 45,  "std": 20},
    "Entertainment": {"merchants": ["Netflix", "Steam", "Cineplex", "Spotify"],          "mean": 20,  "std": 10},
    "Utilities":     {"merchants": ["Nova Scotia Power", "Bell", "Rogers", "City Water"], "mean": 100, "std": 25},
    "Health":        {"merchants": ["Shoppers Drug Mart", "Gym Membership", "Dentist"],  "mean": 60,  "std": 40},
    "Shopping":      {"merchants": ["Amazon", "H&M", "Best Buy", "IKEA"],               "mean": 75,  "std": 50},
    "Income":        {"merchants": ["Employer Payroll", "Freelance Transfer", "E-Transfer"], "mean": 2200, "std": 200},
}

def generate_transactions(months=12):
    records = []
    start = datetime.today() - timedelta(days=months * 30)

    for day_offset in range(months * 30):
        date = start + timedelta(days=day_offset)

        # ~4 transactions per day on average
        n = np.random.poisson(4)
        for _ in range(n):
            cat = random.choices(
                list(CATEGORIES.keys()),
                weights=[15, 12, 10, 8, 5, 5, 10, 3],  # spending more likely than income
                k=1
            )[0]
            cfg = CATEGORIES[cat]
            amount = max(1, np.random.normal(cfg["mean"], cfg["std"]))
            merchant = random.choice(cfg["merchants"])

            # Inject a few anomalies
            is_anomaly = random.random() < 0.02
            if is_anomaly:
                amount *= random.uniform(3, 6)

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "merchant": merchant,
                "category": cat,
                "amount": round(amount, 2),
                "type": "credit" if cat == "Income" else "debit",
                "is_anomaly": is_anomaly,
            })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.index.name = "transaction_id"
    return df


if __name__ == "__main__":
    df = generate_transactions(months=12)
    df.to_csv("data/raw/transactions.csv")
    print(f"✅ Generated {len(df)} transactions → data/raw/transactions.csv")
    print(df.head())
