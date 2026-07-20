#!/usr/bin/env python3
"""Train LSTM streams, ensemble, and meta-learner; log to model_runs + predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.database import get_engine
from equity_compass.training.data import load_features, split_ranges
from equity_compass.training.lstm_models import (
    LSTM_STREAM_FEATURES,
    LSTM_TRAIN_ORDER,
    eval_predictions,
    predict_lstm_avg_ensemble,
    predict_xgboost_meta,
    train_and_predict_stream,
)
from equity_compass.training.registry import ensure_model_tables, evaluate_and_save


def train_one(
    engine,
    ticker: str,
    model_type: str,
    df,
    ranges,
    stream_preds: dict,
    eval_split: str,
) -> tuple[str, dict] | None:
    if model_type in LSTM_STREAM_FEATURES:
        print(f"  Training {model_type}...", flush=True)
        full_preds = train_and_predict_stream(df, ticker, model_type)
        stream_preds[model_type] = full_preds
        preds = eval_predictions(full_preds)
        notes = full_preds.attrs.get("tuning_notes", "")
        metrics = evaluate_and_save(
            engine,
            ticker,
            model_type,
            preds,
            ranges,
            eval_split=eval_split,
            notes=notes,
        )
        print(
            f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}"
        )
        return model_type, metrics

    if model_type == "lstm_avg_ensemble":
        print(f"  Training {model_type}...", flush=True)
        preds = predict_lstm_avg_ensemble(df, ticker, stream_preds)
        notes = preds.attrs.get("tuning_notes", "")
        metrics = evaluate_and_save(
            engine, ticker, model_type, preds, ranges,
            eval_split=eval_split, notes=notes,
        )
        print(
            f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}"
        )
        return model_type, metrics

    if model_type == "xgboost_meta":
        print(f"  Training {model_type}...", flush=True)
        preds = predict_xgboost_meta(df, ticker, stream_preds)
        notes = preds.attrs.get("tuning_notes", "")
        metrics = evaluate_and_save(
            engine, ticker, model_type, preds, ranges,
            eval_split=eval_split, notes=notes,
        )
        print(
            f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}"
        )
        return model_type, metrics

    print(f"  SKIP {model_type}: unknown model")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM and ensemble models.")
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
        models = LSTM_TRAIN_ORDER
    else:
        models = [m.strip() for m in args.models.split(",")]

    print(f"Training LSTM models for {ticker} ({len(df)} feature rows)")
    print(f"Date ranges: train={ranges['train']} val={ranges['val']} test={ranges['test']}")
    print(f"Metrics evaluated on: {args.eval_split}\n")

    stream_preds: dict = {}
    results = []
    for model_type in models:
        try:
            outcome = train_one(
                engine, ticker, model_type, df, ranges, stream_preds, args.eval_split
            )
            if outcome:
                results.append(outcome)
        except Exception as exc:
            print(f"  FAILED {model_type}: {exc}")

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
