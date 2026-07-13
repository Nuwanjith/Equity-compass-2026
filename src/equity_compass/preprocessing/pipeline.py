from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.engine import Engine

from equity_compass.config import (
    BOND_START,
    FEATURES_TO_SCALE,
    FORECAST_HORIZON,
    PIPELINE_VERSION,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from equity_compass.database import delete_ticker_rows


def extract_data(engine: Engine, ticker: str) -> dict[str, pd.DataFrame]:
    prices = pd.read_sql(
        """
        SELECT trade_date, open_price, high_price, low_price,
               close_price, share_volume
        FROM raw_stock_prices
        WHERE ticker = %(ticker)s
        ORDER BY trade_date
        """,
        engine,
        params={"ticker": ticker},
        parse_dates=["trade_date"],
    )

    forex = pd.read_sql(
        """
        SELECT rate_date, close_price AS forex_close
        FROM raw_forex_rates
        ORDER BY rate_date
        """,
        engine,
        parse_dates=["rate_date"],
    )

    bonds = pd.read_sql(
        """
        SELECT rate_date, price AS bond_rate
        FROM raw_bond_rates
        ORDER BY rate_date
        """,
        engine,
        parse_dates=["rate_date"],
    )

    fundamentals = pd.read_sql(
        """
        SELECT report_date, eps, nav_per_share
        FROM raw_fundamentals
        WHERE ticker = %(ticker)s
        ORDER BY report_date
        """,
        engine,
        params={"ticker": ticker},
        parse_dates=["report_date"],
    )

    return {
        "prices": prices,
        "forex": forex,
        "bonds": bonds,
        "fundamentals": fundamentals,
    }


def align_calendars(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["prices"].rename(columns={"trade_date": "feature_date"})
    df = df.set_index("feature_date").sort_index()

    forex = data["forex"].set_index("rate_date").sort_index()
    bonds = data["bonds"].set_index("rate_date").sort_index()
    fund = data["fundamentals"].set_index("report_date").sort_index()

    df = df.join(forex, how="left")
    df["forex_close"] = df["forex_close"].ffill()

    df = df.join(bonds, how="left")
    df["bond_rate"] = df["bond_rate"].ffill()

    df = df.join(fund, how="left")
    df["eps"] = df["eps"].ffill()
    df["nav_per_share"] = df["nav_per_share"].ffill()

    return df


def get_effective_start(df: pd.DataFrame) -> pd.Timestamp:
    bond_start = pd.Timestamp(BOND_START)
    first_eps = df["eps"].first_valid_index()
    candidates = [df.index.min(), bond_start]
    if first_eps is not None:
        candidates.append(pd.Timestamp(first_eps))
    return max(candidates)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close_price"] / out["close_price"].shift(1))
    out["rolling_ma_20"] = out["close_price"].rolling(20).mean()
    out["rolling_std_20"] = out["close_price"].rolling(20).std()
    out["forex_log_return"] = np.log(out["forex_close"] / out["forex_close"].shift(1))
    return out.dropna(subset=["rolling_ma_20", "log_return", "forex_log_return"])


def create_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    out = df.copy()
    out["target_close_30d"] = out["close_price"].shift(-horizon)
    out["target_date_30d"] = pd.Series(out.index, index=out.index).shift(-horizon).values
    return out.dropna(subset=["target_close_30d"])


def assign_split(n: int) -> list[str]:
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    return (
        ["train"] * train_end
        + ["val"] * (val_end - train_end)
        + ["test"] * (n - val_end)
    )


def fit_and_apply_scalers(
    df: pd.DataFrame, ticker: str
) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    train_df = out[out["split_flag"] == "train"]
    scaler_records: list[dict] = []

    for raw_col, scaled_col in FEATURES_TO_SCALE.items():
        train_vals = train_df[[raw_col]].dropna()
        if train_vals.empty:
            out[scaled_col] = np.nan
            continue

        scaler = MinMaxScaler()
        scaler.fit(train_vals)

        col_values = out[[raw_col]].copy()
        fill_value = float(train_vals[raw_col].mean())
        scaled = scaler.transform(col_values.fillna(fill_value))
        out[scaled_col] = scaled.flatten()

        scaler_records.append(
            {
                "ticker": ticker,
                "feature_name": raw_col,
                "scaler_type": "minmax",
                "param_min": float(scaler.data_min_[0]),
                "param_max": float(scaler.data_max_[0]),
                "param_mean": None,
                "param_std": None,
                "fitted_on": train_df.index.max().date(),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    return out, scaler_records


def build_features(engine: Engine, ticker: str) -> tuple[pd.DataFrame, list[dict]]:
    raw = extract_data(engine, ticker)
    df = align_calendars(raw)
    df = df.loc[get_effective_start(df) :].copy()
    df = engineer_features(df)
    df = create_target(df)
    df["split_flag"] = assign_split(len(df))
    return fit_and_apply_scalers(df, ticker)


FEATURE_COLUMNS = [
    "ticker",
    "feature_date",
    "close_price",
    "open_price",
    "high_price",
    "low_price",
    "share_volume",
    "eps",
    "nav_per_share",
    "bond_rate",
    "forex_close",
    "log_return",
    "rolling_ma_20",
    "rolling_std_20",
    "forex_log_return",
    "close_scaled",
    "eps_scaled",
    "bond_rate_scaled",
    "forex_close_scaled",
    "target_close_30d_scaled",
    "target_close_30d",
    "target_date_30d",
    "split_flag",
]


def write_to_db(
    engine: Engine,
    ticker: str,
    df: pd.DataFrame,
    scaler_records: list[dict],
    *,
    replace: bool = True,
) -> int:
    if replace:
        delete_ticker_rows(engine, ticker)

    out = df.reset_index()
    if "feature_date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "feature_date"})
    out["ticker"] = ticker
    out["target_date_30d"] = pd.to_datetime(out["target_date_30d"]).dt.date

    out[FEATURE_COLUMNS].to_sql(
        "features_daily", engine, if_exists="append", index=False
    )

    if scaler_records:
        pd.DataFrame(scaler_records).to_sql(
            "scaler_params", engine, if_exists="append", index=False
        )

    return len(out)


def run_preprocessing(
    engine: Engine, ticker: str, *, replace: bool = True
) -> dict:
    df, scaler_records = build_features(engine, ticker)
    row_count = write_to_db(engine, ticker, df, scaler_records, replace=replace)

    split_counts = df["split_flag"].value_counts().to_dict()
    return {
        "ticker": ticker,
        "rows_written": row_count,
        "scalers_written": len(scaler_records),
        "date_range": (df.index.min().date(), df.index.max().date()),
        "split_counts": split_counts,
    }
