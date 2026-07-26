#!/usr/bin/env python3
"""Generate genuine future (out-of-sample) forecasts by reusing trained models.

Fills the ~30-trading-day gap between the last realised price and the latest
30-day-ahead forecast, so the web dashboard's prediction line extends past the
end of the historical price line. Reuses saved LSTM + xgboost_meta artifacts
(no retraining) and writes rows to `predictions` with split_flag='future'.

Usage:
    python scripts/forecast_future.py                 # all configured tickers
    python scripts/forecast_future.py --ticker DIPD,TYRE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.config import TICKERS
from equity_compass.database import get_engine
from equity_compass.training.forecast import (
    ensure_future_split_enum,
    forecast_future,
    save_future_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate future forecasts.")
    parser.add_argument(
        "--ticker",
        default="DIPD,TYRE",
        help="Ticker symbol, comma-separated list, or 'all' (default: DIPD,TYRE)",
    )
    args = parser.parse_args()

    tickers = (
        TICKERS
        if args.ticker == "all"
        else [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    )

    engine = get_engine()
    ensure_future_split_enum(engine)

    print(f"Forecasting future prices for {len(tickers)} ticker(s)...\n")
    for ticker in tickers:
        try:
            result = forecast_future(engine, ticker)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ticker}: FAILED — {exc}")
            continue
        if result is None:
            print(f"  {ticker}: SKIPPED — no models/features or no tail to forecast")
            continue
        n = save_future_predictions(engine, ticker, result)
        preds = result.predictions
        print(
            f"  {ticker}: {n} future rows, "
            f"target dates {preds['target_date'].min()} → {preds['target_date'].max()}, "
            f"price {preds['predicted_price'].min():.2f} → {preds['predicted_price'].max():.2f}"
        )
        print(f"      {result.notes}")

    print("\nDone.")


if __name__ == "__main__":
    main()
