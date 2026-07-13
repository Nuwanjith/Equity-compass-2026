#!/usr/bin/env python3
"""Run preprocessing: raw MySQL tables → features_daily + scaler_params."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.config import TICKERS
from equity_compass.database import get_engine
from equity_compass.preprocessing.pipeline import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw equity data into features_daily."
    )
    parser.add_argument(
        "--ticker",
        default="TYRE",
        help="Ticker to preprocess (default: TYRE). Use ALL for every ticker.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Append without deleting existing rows for the ticker.",
    )
    args = parser.parse_args()

    engine = get_engine()
    replace = not args.no_replace

    if args.ticker.upper() == "ALL":
        tickers = TICKERS
    else:
        tickers = [args.ticker.upper()]

    for ticker in tickers:
        print(f"Preprocessing {ticker}...")
        try:
            result = run_preprocessing(engine, ticker, replace=replace)
            print(
                f"  rows={result['rows_written']}  "
                f"scalers={result['scalers_written']}  "
                f"dates={result['date_range'][0]} → {result['date_range'][1]}  "
                f"splits={result['split_counts']}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            raise

    print("Done.")


if __name__ == "__main__":
    main()
