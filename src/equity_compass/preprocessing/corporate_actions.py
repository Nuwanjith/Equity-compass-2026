"""Back-adjustment for share subdivisions and similar corporate actions."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from equity_compass.config import CORPORATE_ACTIONS


class SubdivisionAction(TypedDict, total=False):
    effective_date: str
    ratio: float
    transition_through: str
    transition_price_threshold: float


def get_subdivision(ticker: str) -> SubdivisionAction | None:
    action = CORPORATE_ACTIONS.get(ticker.upper(), {}).get("subdivision")
    return action  # type: ignore[return-value]


def apply_subdivision_adjustments(
    ticker: str,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-adjust pre-split prices, volume, EPS, and NAV to post-split basis."""
    action = get_subdivision(ticker)
    if not action:
        return prices, fundamentals

    ratio = float(action["ratio"])
    effective = pd.Timestamp(action["effective_date"])
    prices = prices.copy()
    fundamentals = fundamentals.copy()

    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    pre_split = prices["trade_date"] < effective
    for col in price_cols:
        prices.loc[pre_split, col] = prices.loc[pre_split, col] / ratio
    prices.loc[pre_split, "share_volume"] = prices.loc[pre_split, "share_volume"] * ratio

    transition_through = action.get("transition_through")
    threshold = float(action.get("transition_price_threshold", 100.0))
    if transition_through:
        transition_end = pd.Timestamp(transition_through)
        transition = (
            (prices["trade_date"] >= effective)
            & (prices["trade_date"] <= transition_end)
            & (prices["close_price"] > threshold)
        )
        for col in price_cols:
            prices.loc[transition, col] = prices.loc[transition, col] / ratio
        prices.loc[transition, "share_volume"] = (
            prices.loc[transition, "share_volume"] * ratio
        )

    if not fundamentals.empty:
        pre_split_fund = fundamentals["report_date"] < effective
        for col in ("eps", "nav_per_share"):
            fundamentals.loc[pre_split_fund, col] = (
                fundamentals.loc[pre_split_fund, col] / ratio
            )

    return prices, fundamentals
