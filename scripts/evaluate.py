#!/usr/bin/env python3
"""Phase 6 — generate evaluation reports, tables, and charts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_compass.config import PROJECT_ROOT as CFG_ROOT
from equity_compass.database import get_engine
from equity_compass.evaluation.report import generate_summary_report, generate_ticker_report


def _trained_tickers(engine) -> list[str]:
    from sqlalchemy import text

    rows = engine.connect().execute(
        text("SELECT DISTINCT ticker FROM model_runs ORDER BY ticker")
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 evaluation reports.")
    parser.add_argument(
        "--ticker",
        default="all",
        help="Ticker symbol, comma-separated list, or 'all' (default: all trained tickers)",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "val", "test"],
        help="Split used for monthly error analysis (default: test)",
    )
    parser.add_argument(
        "--output",
        default=str(CFG_ROOT / "reports"),
        help="Output root directory (default: reports/)",
    )
    args = parser.parse_args()

    engine = get_engine()
    output_root = Path(args.output)

    if args.ticker == "all":
        tickers = _trained_tickers(engine)
    else:
        tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    if not tickers:
        print("No tickers with model_runs found. Train models first.")
        sys.exit(1)

    print(f"Phase 6 evaluation — split={args.eval_split}, tickers={tickers}\n")

    for ticker in tickers:
        try:
            out = generate_ticker_report(
                engine,
                ticker,
                eval_split=args.eval_split,
                output_dir=output_root / ticker,
            )
            print(f"  {ticker}: {out / 'report.md'}")
        except Exception as exc:
            print(f"  {ticker}: FAILED — {exc}")

    summary_dir = generate_summary_report(
        engine,
        tickers,
        eval_split=args.eval_split,
        output_dir=output_root / "summary",
    )
    print(f"\nCross-ticker summary: {summary_dir / 'summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
