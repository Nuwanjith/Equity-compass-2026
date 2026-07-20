"""Load model runs and predictions from the database."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_model_runs(engine: Engine, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT run_id, ticker, model_type, trained_at,
               train_from, train_to, val_from, val_to, test_from, test_to,
               mae, rmse, mape, r2, directional_accuracy, notes
        FROM model_runs
        WHERE ticker = %(ticker)s
        ORDER BY mae
        """,
        engine,
        params={"ticker": ticker},
        parse_dates=["trained_at"],
    )
    return df


def load_predictions(engine: Engine, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT p.run_id, m.model_type, p.ticker, p.predicted_at, p.target_date,
               p.predicted_price, p.actual_price, p.split_flag
        FROM predictions p
        JOIN model_runs m ON m.run_id = p.run_id
        WHERE p.ticker = %(ticker)s
        ORDER BY m.model_type, p.predicted_at
        """,
        engine,
        params={"ticker": ticker},
        parse_dates=["predicted_at", "target_date"],
    )
    df["predicted_price"] = df["predicted_price"].astype(float)
    df["actual_price"] = df["actual_price"].astype(float)
    return df
