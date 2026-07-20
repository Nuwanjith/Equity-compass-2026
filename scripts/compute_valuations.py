#!/usr/bin/env python3
"""Compute NAV-based, EPS-based, and Graham Number valuations for tickers
and upsert them into the `valuations` table, for use by the web frontend's
Valuation panel (`stock_valuations.php`).

Usage:
    python scripts/compute_valuations.py               # all tickers in config.TICKERS
    python scripts/compute_valuations.py --ticker TYRE,DIPD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.config import TICKERS
from equity_compass.database import get_engine
from equity_compass.valuation import (
    compute_valuation,
    ensure_valuations_table,
    save_valuation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fundamentals-based valuations.")
    parser.add_argument(
        "--ticker",
        default="all",
        help="Ticker symbol, comma-separated list, or 'all' (default: all configured tickers)",
    )
    args = parser.parse_args()

    tickers = (
        TICKERS
        if args.ticker == "all"
        else [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    )

    engine = get_engine()
    ensure_valuations_table(engine)

    print(f"Computing valuations for {len(tickers)} ticker(s)...\n")
    for ticker in tickers:
        result = compute_valuation(engine, ticker)
        if result is None:
            print(f"  {ticker}: SKIPPED — no fundamentals data")
            continue
        save_valuation(engine, result)
        print(
            f"  {ticker} [{result.quarter}]: "
            f"NAV={result.nav_valuation}, "
            f"EPS={result.eps_valuation}, "
            f"Graham={result.graham_valuation}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
