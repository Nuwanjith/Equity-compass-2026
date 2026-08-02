"""Fundamentals-based fair-value estimates (NAV, EPS-multiple, Graham Number).

These are simple, point-in-time valuation heuristics computed from the latest
reported quarterly fundamentals in `raw_fundamentals` — independent of the
LSTM/XGBoost price-forecast models. They power the "Valuation" panel in the
web frontend (`stock_valuations.php`).

Calibration note: the legacy system (`Equity-Compass-Web`, pre-migration) had
exactly one historical valuation row with enough precision to reverse-engineer
against (TYRE, quarter 2025-Q1, computed 2025-06-30): NAV-based=94.33,
EPS-based=93.50, Graham=138.90. Using that ticker/quarter's fundamentals
(latest NAV/share=94.44, trailing-twelve-month EPS=9.03, i.e. the sum of the
four most recent reported quarters):
  - Graham Number = sqrt(22.5 * ttm_eps * nav_per_share)   -> 138.54 (~0.3% off)
    This is the unmodified, textbook Benjamin Graham formula — no fudge factor
    needed.
  - NAV-based = nav_per_share, unmodified                  -> 94.44  (~0.1% off)
  - EPS-based = ttm_eps * EPS_FAIR_PE_MULTIPLE              -> 9.03 * 10.35 = 93.46
    (~0.04% off) — VALUATION_EPS_FAIR_PE below is calibrated to this.
The small residual differences are consistent with rounding in the
originally-reported (2-dp) quarterly EPS figures and are treated as noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# "Fair" P/E multiple applied to trailing-twelve-month EPS. See module
# docstring for how this was calibrated.
VALUATION_EPS_FAIR_PE = 10.35

# Number of trailing quarters summed to build a TTM EPS figure.
TTM_QUARTERS = 4


@dataclass
class ValuationResult:
    ticker: str
    quarter: str
    report_date: pd.Timestamp
    nav_per_share: float | None
    ttm_eps: float | None
    nav_valuation: float | None
    eps_valuation: float | None
    graham_valuation: float | None


def _fiscal_quarter_label(fiscal_year: int, quarter: int) -> str:
    return f"{int(fiscal_year)}-Q{int(quarter)}"


def compute_valuation(engine: Engine, ticker: str) -> ValuationResult | None:
    """Compute NAV-based, EPS-based, and Graham Number valuations for `ticker`.

    Uses the most recently reported quarter in `raw_fundamentals` as the
    as-of point, and the trailing `TTM_QUARTERS` quarters' EPS (summed) as
    the annualized earnings figure. Returns None if no fundamentals exist.
    """
    fundamentals = pd.read_sql(
        """
        SELECT fiscal_year, quarter, report_date, eps, nav_per_share
        FROM raw_fundamentals
        WHERE ticker = %(ticker)s
        ORDER BY report_date
        """,
        engine,
        params={"ticker": ticker},
    )
    if fundamentals.empty:
        return None

    latest = fundamentals.iloc[-1]
    ttm_window = fundamentals.tail(TTM_QUARTERS)
    ttm_eps = (
        float(ttm_window["eps"].sum())
        if ttm_window["eps"].notna().all() and len(ttm_window) == TTM_QUARTERS
        else None
    )
    nav_per_share = (
        float(latest["nav_per_share"]) if pd.notna(latest["nav_per_share"]) else None
    )

    nav_valuation = nav_per_share
    eps_valuation = ttm_eps * VALUATION_EPS_FAIR_PE if ttm_eps is not None else None
    graham_valuation = (
        math.sqrt(22.5 * ttm_eps * nav_per_share)
        if ttm_eps is not None and nav_per_share is not None
        and ttm_eps > 0 and nav_per_share > 0
        else None
    )

    return ValuationResult(
        ticker=ticker,
        quarter=_fiscal_quarter_label(latest["fiscal_year"], latest["quarter"]),
        report_date=pd.Timestamp(latest["report_date"]),
        nav_per_share=nav_per_share,
        ttm_eps=ttm_eps,
        nav_valuation=nav_valuation,
        eps_valuation=eps_valuation,
        graham_valuation=graham_valuation,
    )


def ensure_valuations_table(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS valuations (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        quarter VARCHAR(10) NOT NULL,
        report_date DATE NULL,
        nav_per_share DECIMAL(12,4) NULL,
        ttm_eps DECIMAL(12,4) NULL,
        nav_valuation DECIMAL(12,2) NULL,
        eps_valuation DECIMAL(12,2) NULL,
        graham_valuation DECIMAL(12,2) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uniq_ticker_quarter (ticker, quarter)
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def save_valuation(engine: Engine, result: ValuationResult) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO valuations (
                    ticker, quarter, report_date, nav_per_share, ttm_eps,
                    nav_valuation, eps_valuation, graham_valuation
                ) VALUES (
                    :ticker, :quarter, :report_date, :nav_per_share, :ttm_eps,
                    :nav_valuation, :eps_valuation, :graham_valuation
                )
                ON DUPLICATE KEY UPDATE
                    report_date = VALUES(report_date),
                    nav_per_share = VALUES(nav_per_share),
                    ttm_eps = VALUES(ttm_eps),
                    nav_valuation = VALUES(nav_valuation),
                    eps_valuation = VALUES(eps_valuation),
                    graham_valuation = VALUES(graham_valuation)
                """
            ),
            {
                "ticker": result.ticker,
                "quarter": result.quarter,
                "report_date": result.report_date.date(),
                "nav_per_share": result.nav_per_share,
                "ttm_eps": result.ttm_eps,
                "nav_valuation": result.nav_valuation,
                "eps_valuation": result.eps_valuation,
                "graham_valuation": result.graham_valuation,
            },
        )
