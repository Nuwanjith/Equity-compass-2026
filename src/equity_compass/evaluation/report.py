"""Build comparison tables and markdown reports for Phase 6 evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_compass.config import FORECAST_HORIZON, PROJECT_ROOT
from equity_compass.evaluation.charts import save_ticker_charts
from equity_compass.evaluation.loaders import load_model_runs, load_predictions
from equity_compass.training.metrics import compute_metrics

MODEL_ORDER = [
    "naive",
    "linear_regression",
    "arima",
    "arimax",
    "xgboost_tabular",
    "lstm_univariate",
    "lstm_fundamental",
    "lstm_macro",
    "lstm_forex",
    "lstm_avg_ensemble",
    "xgboost_meta",
]

BASELINE_MODELS = {
    "naive",
    "linear_regression",
    "arima",
    "arimax",
    "xgboost_tabular",
}
LSTM_STREAM_MODELS = {
    "lstm_univariate",
    "lstm_fundamental",
    "lstm_macro",
    "lstm_forex",
}
ENSEMBLE_MODELS = {"lstm_avg_ensemble", "xgboost_meta"}


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_type",
        "mae",
        "rmse",
        "mape",
        "r2",
        "directional_accuracy",
        "test_from",
        "test_to",
    ]
    out = runs[cols].copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out.sort_values("mae").reset_index(drop=True)


def category_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [
        ("Best baseline", BASELINE_MODELS),
        ("Best LSTM stream", LSTM_STREAM_MODELS),
        ("Best ensemble", ENSEMBLE_MODELS),
        ("Overall best", set(runs["model_type"])),
    ]
    for label, members in groups:
        subset = runs[runs["model_type"].isin(members)]
        if subset.empty:
            continue
        best = subset.sort_values("mae").iloc[0]
        rows.append(
            {
                "category": label,
                "model_type": best["model_type"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "r2": best["r2"],
            }
        )
    return pd.DataFrame(rows)


def lstm_ablation_table(runs: pd.DataFrame) -> pd.DataFrame:
    lstm = runs[runs["model_type"].isin(LSTM_STREAM_MODELS | ENSEMBLE_MODELS)].copy()
    if lstm.empty:
        return lstm
    baseline_best = runs[runs["model_type"].isin(BASELINE_MODELS)]["mae"].min()
    lstm = lstm.sort_values("mae").reset_index(drop=True)
    lstm["rank"] = np.arange(1, len(lstm) + 1)
    lstm["vs_best_baseline_mae"] = lstm["mae"] - baseline_best
    lstm["vs_best_baseline_pct"] = (
        (lstm["mae"] - baseline_best) / baseline_best * 100
    ).round(2)
    return lstm[
        [
            "rank",
            "model_type",
            "mae",
            "rmse",
            "mape",
            "r2",
            "vs_best_baseline_mae",
            "vs_best_baseline_pct",
        ]
    ]


def improvement_vs_baseline(runs: pd.DataFrame) -> pd.DataFrame:
    best_baseline = runs[runs["model_type"].isin(BASELINE_MODELS)].sort_values("mae").iloc[0]
    best_overall = runs.sort_values("mae").iloc[0]
    mae_delta = best_overall["mae"] - best_baseline["mae"]
    mae_pct = mae_delta / best_baseline["mae"] * 100
    return pd.DataFrame(
        [
            {
                "best_baseline": best_baseline["model_type"],
                "best_baseline_mae": best_baseline["mae"],
                "best_model": best_overall["model_type"],
                "best_model_mae": best_overall["mae"],
                "mae_improvement": -mae_delta,
                "mae_improvement_pct": -mae_pct,
            }
        ]
    )


def monthly_error_table(
    predictions: pd.DataFrame,
    model_type: str,
    *,
    eval_split: str = "test",
) -> pd.DataFrame:
    subset = predictions[
        (predictions["model_type"] == model_type)
        & (predictions["split_flag"] == eval_split)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["month"] = subset["predicted_at"].dt.to_period("M").astype(str)
    rows = []
    for month, group in subset.groupby("month", sort=True):
        y_true = group["actual_price"].values
        y_pred = group["predicted_price"].values
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "month": month,
                "n_predictions": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def split_metrics_table(
    predictions: pd.DataFrame,
    model_type: str,
) -> pd.DataFrame:
    rows = []
    subset = predictions[predictions["model_type"] == model_type]
    for split_flag, group in subset.groupby("split_flag"):
        metrics = compute_metrics(
            group["actual_price"].values,
            group["predicted_price"].values,
        )
        rows.append({"split_flag": split_flag, "n": len(group), **metrics})
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No data._\n"
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda x: format(x, float_fmt))
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_report(
    path: Path,
    ticker: str,
    runs: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    eval_split: str,
    best_model: str,
) -> None:
    best = runs.sort_values("mae").iloc[0]
    test_from = runs["test_from"].dropna().iloc[0] if not runs.empty else None
    test_to = runs["test_to"].dropna().iloc[0] if not runs.empty else None

    lines = [
        f"# Phase 6 Evaluation — {ticker}",
        "",
        f"**Forecast horizon:** {FORECAST_HORIZON} trading days  ",
        f"**Metrics split:** `{eval_split}`  ",
        f"**Test period:** {test_from} → {test_to}  ",
        f"**Best model:** `{best['model_type']}` (MAE **{best['mae']:.4f}** Rs)",
        "",
        "## Model leaderboard",
        "",
        _df_to_markdown(tables["comparison"]),
        "",
        "## Category summary",
        "",
        _df_to_markdown(tables["category"]),
        "",
        "## LSTM ablation",
        "",
        _df_to_markdown(tables["lstm_ablation"]),
        "",
        "## Improvement vs best baseline",
        "",
        _df_to_markdown(tables["improvement"], float_fmt=".2f"),
        "",
        f"## Monthly error — `{best_model}` ({eval_split})",
        "",
        _df_to_markdown(tables["monthly"]),
        "",
        f"## Split metrics — `{best_model}`",
        "",
        _df_to_markdown(tables["split_metrics"]),
        "",
        "## Charts",
        "",
        "- `mae_comparison.png` — MAE by model",
        "- `lstm_ablation.png` — LSTM stream comparison",
        f"- `predicted_vs_actual_{best_model}.png` — best model vs actual",
        f"- `monthly_mae_{best_model}.png` — monthly MAE trend",
        "",
    ]
    path.write_text("\n".join(lines))


def generate_ticker_report(
    engine: Engine,
    ticker: str,
    *,
    eval_split: str = "test",
    output_dir: Path | None = None,
) -> Path:
    """Generate CSV tables, charts, and markdown report for one ticker."""
    ticker = ticker.upper()
    out_dir = output_dir or (PROJECT_ROOT / "reports" / ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_model_runs(engine, ticker)
    if runs.empty:
        raise ValueError(f"No model_runs found for {ticker}")

    predictions = load_predictions(engine, ticker)
    runs = runs.sort_values("mae").reset_index(drop=True)
    best_model = runs.iloc[0]["model_type"]

    tables = {
        "comparison": comparison_table(runs),
        "category": category_summary(runs),
        "lstm_ablation": lstm_ablation_table(runs),
        "improvement": improvement_vs_baseline(runs),
        "monthly": monthly_error_table(predictions, best_model, eval_split=eval_split),
        "split_metrics": split_metrics_table(predictions, best_model),
    }

    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)

    save_ticker_charts(
        out_dir,
        runs,
        predictions,
        best_model=best_model,
        eval_split=eval_split,
    )

    _write_markdown_report(
        out_dir / "report.md",
        ticker,
        runs,
        tables,
        eval_split=eval_split,
        best_model=best_model,
    )
    return out_dir


def generate_summary_report(
    engine: Engine,
    tickers: list[str],
    *,
    eval_split: str = "test",
    output_dir: Path | None = None,
) -> Path:
    """Cross-ticker summary for thesis comparison."""
    out_dir = output_dir or (PROJECT_ROOT / "reports" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in tickers:
        runs = load_model_runs(engine, ticker.upper())
        if runs.empty:
            continue
        best = runs.sort_values("mae").iloc[0]
        best_base = runs[runs["model_type"].isin(BASELINE_MODELS)].sort_values("mae").iloc[0]
        rows.append(
            {
                "ticker": ticker.upper(),
                "best_model": best["model_type"],
                "best_mae": best["mae"],
                "best_baseline": best_base["model_type"],
                "best_baseline_mae": best_base["mae"],
                "improvement_pct": (best_base["mae"] - best["mae"]) / best_base["mae"] * 100,
                "test_from": best["test_from"],
                "test_to": best["test_to"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cross_ticker_summary.csv", index=False)

    lines = [
        "# Phase 6 Cross-Ticker Summary",
        "",
        f"**Metrics split:** `{eval_split}`",
        "",
        _df_to_markdown(summary),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))
    return out_dir
