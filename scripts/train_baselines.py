#!/usr/bin/env python3
"""Train baseline models and log results to model_runs + predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.database import get_engine
from equity_compass.training.baselines import BASELINE_MODELS
from equity_compass.training.data import load_features, split_ranges
from equity_compass.training.registry import ensure_model_tables, evaluate_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline forecasting models.")
    parser.add_argument("--ticker", default="TYRE", help="Ticker symbol (default: TYRE)")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all' (default: all)",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test"],
        help="Split used for reported metrics (default: test)",
    )
    args = parser.parse_args()

    engine = get_engine()
    ensure_model_tables(engine)

    ticker = args.ticker.upper()
    df = load_features(engine, ticker)
    ranges = split_ranges(df)

    if args.models == "all":
        models = list(BASELINE_MODELS.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]

    print(f"Training baselines for {ticker} ({len(df)} feature rows)")
    print(f"Date ranges: train={ranges['train']} val={ranges['val']} test={ranges['test']}")
    print(f"Metrics evaluated on: {args.eval_split}\n")

    results = []
    for model_type in models:
        if model_type not in BASELINE_MODELS:
            print(f"  SKIP {model_type}: unknown model")
            continue

        print(f"  Training {model_type}...", end=" ", flush=True)
        try:
            predict_fn = BASELINE_MODELS[model_type]
            predictions_df = predict_fn(df)
            tuning_notes = predictions_df.attrs.get("tuning_notes")
            base_note = f"Evaluated on {args.eval_split} split"
            notes = f"{base_note}; {tuning_notes}" if tuning_notes else base_note
            metrics = evaluate_and_save(
                engine,
                ticker,
                model_type,
                predictions_df,
                ranges,
                eval_split=args.eval_split,
                notes=notes,
            )
            results.append((model_type, metrics))
            if tuning_notes:
                print(f"\n    {tuning_notes}")
            print(
                f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                f"MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        print("\n--- Summary (test metrics) ---")
        print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8}")
        for model_type, m in results:
            print(
                f"{model_type:<22} {m['mae']:8.4f} {m['rmse']:8.4f} "
                f"{m['mape']:8.2f} {m['r2']:8.4f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
