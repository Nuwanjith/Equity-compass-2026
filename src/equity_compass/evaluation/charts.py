"""Matplotlib charts for Phase 6 evaluation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LSTM_COLORS = {
    "lstm_univariate": "#2563eb",
    "lstm_fundamental": "#16a34a",
    "lstm_macro": "#9333ea",
    "lstm_forex": "#ea580c",
    "lstm_avg_ensemble": "#0891b2",
    "xgboost_meta": "#64748b",
}


def _model_colors(model_types: list[str]) -> list[str]:
    palette = ["#94a3b8", "#475569", "#78716c", "#a8a29e", "#57534e"]
    colors = []
    for i, model in enumerate(model_types):
        if model in LSTM_COLORS:
            colors.append(LSTM_COLORS[model])
        else:
            colors.append(palette[i % len(palette)])
    return colors


def save_mae_comparison(out_dir: Path, runs: pd.DataFrame) -> None:
    ordered = runs.sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.45)))
    colors = _model_colors(ordered["model_type"].tolist())
    ax.barh(ordered["model_type"], ordered["mae"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (Rs)")
    ax.set_title("Model comparison — MAE (lower is better)")
    ax.axvline(ordered["mae"].iloc[0], color="#dc2626", linestyle="--", alpha=0.5, label="Best")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mae_comparison.png", dpi=150)
    plt.close(fig)


def save_lstm_ablation(out_dir: Path, runs: pd.DataFrame) -> None:
    lstm_types = {
        "lstm_univariate",
        "lstm_fundamental",
        "lstm_macro",
        "lstm_forex",
        "lstm_avg_ensemble",
        "xgboost_meta",
    }
    subset = runs[runs["model_type"].isin(lstm_types)].sort_values("mae")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = _model_colors(subset["model_type"].tolist())
    ax.bar(subset["model_type"], subset["mae"], color=colors)
    ax.set_ylabel("Test MAE (Rs)")
    ax.set_title("LSTM ablation — stream and ensemble models")
    ax.tick_params(axis="x", rotation=30)
    best_base = runs[~runs["model_type"].isin(lstm_types)]["mae"].min()
    ax.axhline(best_base, color="#dc2626", linestyle="--", label=f"Best baseline ({best_base:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lstm_ablation.png", dpi=150)
    plt.close(fig)


def save_predicted_vs_actual(
    out_dir: Path,
    predictions: pd.DataFrame,
    model_type: str,
    *,
    eval_split: str = "test",
) -> None:
    subset = predictions[
        (predictions["model_type"] == model_type)
        & (predictions["split_flag"] == eval_split)
    ].sort_values("predicted_at")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        subset["target_date"],
        subset["actual_price"],
        label="Actual (30d target)",
        color="#111827",
        linewidth=1.5,
    )
    ax.plot(
        subset["target_date"],
        subset["predicted_price"],
        label=f"Predicted ({model_type})",
        color="#2563eb",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.set_xlabel("Target date")
    ax.set_ylabel("Price (Rs)")
    ax.set_title(f"Predicted vs actual — {model_type} ({eval_split})")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / f"predicted_vs_actual_{model_type}.png", dpi=150)
    plt.close(fig)


def save_monthly_mae(
    out_dir: Path,
    predictions: pd.DataFrame,
    model_type: str,
    *,
    eval_split: str = "test",
) -> None:
    subset = predictions[
        (predictions["model_type"] == model_type)
        & (predictions["split_flag"] == eval_split)
    ].copy()
    if subset.empty:
        return

    subset["month"] = subset["predicted_at"].dt.to_period("M").astype(str)
    monthly = (
        subset.assign(abs_err=lambda d: (d["actual_price"] - d["predicted_price"]).abs())
        .groupby("month", sort=True)["abs_err"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(monthly.index.astype(str), monthly.values, color="#2563eb")
    ax.set_ylabel("MAE (Rs)")
    ax.set_xlabel("Month (prediction date)")
    ax.set_title(f"Monthly MAE — {model_type} ({eval_split})")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / f"monthly_mae_{model_type}.png", dpi=150)
    plt.close(fig)


def save_ticker_charts(
    out_dir: Path,
    runs: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    best_model: str,
    eval_split: str = "test",
) -> None:
    save_mae_comparison(out_dir, runs)
    save_lstm_ablation(out_dir, runs)
    save_predicted_vs_actual(out_dir, predictions, best_model, eval_split=eval_split)
    save_monthly_mae(out_dir, predictions, best_model, eval_split=eval_split)
