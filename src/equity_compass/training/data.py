from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

TABULAR_FEATURES = [
    "close_price",
    "log_return",
    "rolling_ma_20",
    "rolling_std_20",
    "eps",
    "bond_rate",
    "forex_close",
    "forex_log_return",
]

EXOG_FEATURES = ["forex_close", "bond_rate", "eps"]


def load_features(engine: Engine, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT feature_date, close_price, open_price, high_price, low_price,
               share_volume, eps, nav_per_share, bond_rate, forex_close,
               log_return, rolling_ma_20, rolling_std_20, forex_log_return,
               close_scaled, eps_scaled, bond_rate_scaled, forex_close_scaled,
               target_close_30d, target_date_30d, split_flag
        FROM features_daily
        WHERE ticker = %(ticker)s
        ORDER BY feature_date
        """,
        engine,
        params={"ticker": ticker},
        parse_dates=["feature_date", "target_date_30d"],
    )
    return df


def split_ranges(df: pd.DataFrame) -> dict[str, tuple]:
    ranges = {}
    for flag in ("train", "val", "test"):
        part = df[df["split_flag"] == flag]
        if part.empty:
            ranges[flag] = (None, None)
        else:
            ranges[flag] = (
                part["feature_date"].min().date(),
                part["feature_date"].max().date(),
            )
    return ranges


def get_history_closes(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    allowed = {"train"} if row["split_flag"] == "val" else {"train", "val"}
    mask = (df["feature_date"] <= row["feature_date"]) & (
        df["split_flag"].isin(allowed)
    )
    return df.loc[mask, "close_price"].astype(float)


def get_history_exog(
    df: pd.DataFrame, row: pd.Series, columns: list[str] | None = None
) -> pd.DataFrame:
    cols = columns or EXOG_FEATURES
    allowed = {"train"} if row["split_flag"] == "val" else {"train", "val"}
    mask = (df["feature_date"] <= row["feature_date"]) & (
        df["split_flag"].isin(allowed)
    )
    return df.loc[mask, cols].astype(float)
