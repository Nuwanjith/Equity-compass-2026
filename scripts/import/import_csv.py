import csv
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path("/Users/nuwanjiths/MSC/CODE-2026/Equity-compass-2026")
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.database import get_engine
from equity_compass.preprocessing.pipeline import run_preprocessing

CSV_PATH = PROJECT_ROOT / "DATA/EPS-and-NAV/stock_performance-DIPD.csv"
TICKER = "DIPD"

QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

rows = []
with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
    for rec in csv.DictReader(fh):
        eps_raw = (rec.get("EPS") or "").strip()
        nav_raw = (rec.get("NAV") or "").strip()
        if not eps_raw or not nav_raw:
            continue
        year_str, q_str = rec["Quarter"].split("-")
        fiscal_year = int(year_str)
        quarter = int(q_str)
        month, day = QUARTER_END[quarter]
        rows.append(
            {
                "ticker": TICKER,
                "fiscal_year": fiscal_year,
                "quarter": quarter,
                "report_date": date(fiscal_year, month, day),
                "eps": float(eps_raw),
                "nav_per_share": float(nav_raw),
            }
        )

df = pd.DataFrame(rows)
engine = get_engine()

with engine.begin() as conn:
    deleted = conn.execute(
        text("DELETE FROM raw_fundamentals WHERE ticker = :ticker"),
        {"ticker": TICKER},
    ).rowcount
    df.to_sql("raw_fundamentals", conn, if_exists="append", index=False)

with engine.connect() as conn:
    count = conn.execute(
        text("SELECT COUNT(*) FROM raw_fundamentals WHERE ticker = :ticker"),
        {"ticker": TICKER},
    ).scalar()
    sample = conn.execute(
        text(
            """
            SELECT fiscal_year, quarter, report_date, eps, nav_per_share
            FROM raw_fundamentals
            WHERE ticker = :ticker
            ORDER BY report_date
            LIMIT 3
            """
        ),
        {"ticker": TICKER},
    ).fetchall()
    latest = conn.execute(
        text(
            """
            SELECT fiscal_year, quarter, report_date, eps, nav_per_share
            FROM raw_fundamentals
            WHERE ticker = :ticker
            ORDER BY report_date DESC
            LIMIT 1
            """
        ),
        {"ticker": TICKER},
    ).fetchone()

print(f"Deleted existing DIPD rows: {deleted}")
print(f"Inserted {len(df)} DIPD fundamental rows (now {count} in DB)")
print("First rows:", sample)
print("Latest row:", latest)

print("\nRunning preprocessing for DIPD...")
result = run_preprocessing(engine, TICKER, replace=True)
print(
    f"rows={result['rows_written']}  scalers={result['scalers_written']}  "
    f"dates={result['date_range'][0]} -> {result['date_range'][1]}  "
    f"splits={result['split_counts']}"
)
PY
