"""Out-of-sample ("future") forecasting that REUSES already-trained artifacts.

The training pipeline is a backtest: `create_target` drops every row whose
30-trading-day-ahead close isn't known yet, so no model ever forecasts beyond
the last available price. This module fills that gap. For the most recent
`FORECAST_HORIZON` trading days (which have inputs but no realised target) it:

  1. rebuilds features from raw data (without dropping the target-less tail),
  2. runs each saved LSTM stream (`model.keras` + `scaler.json`),
  3. recombines them into the shrunk val-MAE-weighted `ensemble`,
  4. runs the saved `xgboost_meta` model,
  5. blends `ensemble` + `meta` with the SAME shrinkage rule used in training,

producing genuine future prices (actual_price = NULL) whose target dates land
in the next ~30 business days. Nothing is retrained.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

from equity_compass.config import (
    FEATURES_TO_SCALE,
    FORECAST_HORIZON,
    LSTM_META_STREAMS,
    LSTM_TABULAR_CONTEXT,
)
from equity_compass.preprocessing.pipeline import (
    align_calendars,
    engineer_features,
    extract_data,
    get_effective_start,
)
from equity_compass.training.data import TABULAR_FEATURES, load_features
from equity_compass.training.lstm_models import (
    _prepare_model_df,
    _shrink_weights,
    model_dir,
)
from equity_compass.training.sequences import (
    build_sequences,
    transform_sequences,
    transform_tabular,
)


@dataclass
class ForecastResult:
    ticker: str
    predictions: pd.DataFrame  # feature_date, target_date, predicted_price
    anchor_mode: str
    notes: str


def _load_std_scaler(mean: list[float], scale: list[float]) -> StandardScaler:
    """Reconstruct a fitted StandardScaler from persisted mean/scale."""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=float)
    scaler.scale_ = np.asarray(scale, dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def _load_minmax_params(engine: Engine, ticker: str) -> dict[str, tuple[float, float]]:
    rows = pd.read_sql(
        """
        SELECT feature_name, param_min, param_max
        FROM scaler_params
        WHERE ticker = %(ticker)s AND scaler_type = 'minmax'
        """,
        engine,
        params={"ticker": ticker},
    )
    return {
        r.feature_name: (float(r.param_min), float(r.param_max))
        for r in rows.itertuples()
    }


def build_inference_frame(engine: Engine, ticker: str) -> pd.DataFrame | None:
    """Continuous feature frame = persisted features_daily + the dropped tail.

    Historical rows are taken verbatim from `features_daily` (same splits and
    scaled columns used in training). The most recent target-less rows are
    rebuilt from raw data, MinMax-scaled with the stored training params, and
    flagged `split_flag = 'future'`. Returns None if there is no tail to fill.
    """
    base = load_features(engine, ticker)
    if base.empty:
        return None
    known_max = base["feature_date"].max()

    raw = extract_data(engine, ticker)
    full = align_calendars(raw)
    full = full.loc[get_effective_start(full):].copy()
    full = engineer_features(full)
    full = full.reset_index().rename(columns={"index": "feature_date"})
    if "feature_date" not in full.columns:
        full = full.rename(columns={full.columns[0]: "feature_date"})
    full["feature_date"] = pd.to_datetime(full["feature_date"])

    tail = full[full["feature_date"] > known_max].copy()
    if tail.empty:
        return None

    params = _load_minmax_params(engine, ticker)
    for raw_col, scaled_col in FEATURES_TO_SCALE.items():
        if raw_col == "target_close_30d" or raw_col not in params:
            continue
        mn, mx = params[raw_col]
        rng = (mx - mn) or 1.0
        tail[scaled_col] = (tail[raw_col].astype(float) - mn) / rng

    tail["split_flag"] = "future"
    tail["target_close_30d"] = np.nan
    tail["target_date_30d"] = tail["feature_date"] + pd.tseries.offsets.BusinessDay(
        FORECAST_HORIZON
    )

    for col in base.columns:
        if col not in tail.columns:
            tail[col] = np.nan

    frame = pd.concat([base[base.columns], tail[base.columns]], ignore_index=True)
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame["target_date_30d"] = pd.to_datetime(frame["target_date_30d"])
    return frame.sort_values("feature_date").reset_index(drop=True)


def _stream_predictions(
    model_df: pd.DataFrame,
    anchor_col: str,
    ticker: str,
    stream: str,
    splits: tuple[str, ...] = ("val", "future"),
) -> dict[str, pd.DataFrame]:
    """Run one saved LSTM stream on the requested splits (inference only)."""
    import tensorflow as tf

    sdir = model_dir(ticker, stream)
    cfg = json.loads((sdir / "scaler.json").read_text())
    lookback = int(cfg["lookback"])
    feats = cfg["features"]
    alpha = float(cfg.get("residual_alpha", 1.0))
    seq_scaler = _load_std_scaler(cfg["seq_mean"], cfg["seq_scale"])
    tab_scaler = _load_std_scaler(cfg["tab_mean"], cfg["tab_scale"])
    model = tf.keras.models.load_model(sdir / "model.keras")

    out: dict[str, pd.DataFrame] = {}
    for split in splits:
        batch = build_sequences(
            model_df, feats, lookback=lookback, splits={split},
            tabular_cols=LSTM_TABULAR_CONTEXT, anchor_col=anchor_col,
        )
        if len(batch.x) == 0:
            out[split] = pd.DataFrame(
                columns=["feature_date", "predicted_price", "target_close_30d", "target_date_30d"]
            )
            continue
        x = transform_sequences(batch.x, seq_scaler)
        tab = transform_tabular(batch.tab, tab_scaler)
        model_input = [x, tab] if len(tab) else x
        residual = model.predict(model_input, verbose=0).reshape(-1) * alpha
        price = residual + batch.anchor_close.astype(float)
        out[split] = pd.DataFrame(
            {
                "feature_date": pd.to_datetime(batch.feature_dates),
                "predicted_price": price.astype(float),
                "target_close_30d": batch.y_actual.astype(float),
                "target_date_30d": pd.to_datetime(batch.target_dates),
            }
        ).sort_values("feature_date").reset_index(drop=True)
    return out


def _mae(pred: pd.DataFrame) -> float:
    valid = pred.dropna(subset=["target_close_30d", "predicted_price"])
    if valid.empty:
        return float("inf")
    return float(np.mean(np.abs(valid["target_close_30d"] - valid["predicted_price"])))


def _parse_meta_val_mae(notes: str | None) -> float | None:
    if not notes:
        return None
    m = re.search(r"meta_val_mae=([0-9]*\.?[0-9]+)", notes)
    return float(m.group(1)) if m else None


def _latest_meta_run(engine: Engine, ticker: str) -> tuple[int, str] | None:
    row = pd.read_sql(
        """
        SELECT run_id, notes FROM model_runs
        WHERE ticker = %(ticker)s AND model_type = 'xgboost_meta'
        ORDER BY trained_at DESC LIMIT 1
        """,
        engine,
        params={"ticker": ticker},
    )
    if row.empty:
        return None
    return int(row.iloc[0]["run_id"]), row.iloc[0]["notes"]


def forecast_future(engine: Engine, ticker: str) -> ForecastResult | None:
    """Produce the final blended future forecast for `ticker` (xgboost_meta)."""
    import xgboost as xgb

    frame = build_inference_frame(engine, ticker)
    if frame is None:
        return None

    model_df, anchor_col, anchor_mode = _prepare_model_df(frame, ticker)

    available = [
        s for s in LSTM_META_STREAMS if (model_dir(ticker, s) / "model.keras").exists()
    ]
    if len(available) < 2:
        raise ValueError(f"Need >=2 trained streams to forecast, found {available}")

    stream_out = {
        s: _stream_predictions(model_df, anchor_col, ticker, s) for s in available
    }

    # Per-stream validation MAE -> shrunk inverse-error ensemble weights,
    # reproducing _weighted_ensemble_predictions on live inference.
    val_maes = np.array([_mae(stream_out[s]["val"]) for s in available], dtype=float)
    ens_weights = _shrink_weights(1.0 / (val_maes + 1e-6))

    future_dates = stream_out[available[0]]["future"]["feature_date"].reset_index(drop=True)
    fut_matrix = np.vstack(
        [stream_out[s]["future"]["predicted_price"].to_numpy(dtype=float) for s in available]
    )
    ensemble_future = (ens_weights[:, None] * fut_matrix).sum(axis=0)

    # Ensemble validation MAE (for the ensemble-vs-meta blend weight).
    val_matrix = np.vstack(
        [stream_out[s]["val"]["predicted_price"].to_numpy(dtype=float) for s in available]
    )
    ensemble_val = (ens_weights[:, None] * val_matrix).sum(axis=0)
    val_actual = stream_out[available[0]]["val"]["target_close_30d"].to_numpy(dtype=float)
    ensemble_val_mae = float(np.mean(np.abs(val_actual - ensemble_val)))

    # Meta model on the future rows.
    fut_mask = model_df["split_flag"] == "future"
    fut = model_df.loc[fut_mask, ["feature_date", "target_date_30d"] + LSTM_TABULAR_CONTEXT].copy()
    fut["feature_date"] = pd.to_datetime(fut["feature_date"])
    fut["anchor"] = model_df.loc[fut_mask, anchor_col].astype(float).values
    for s in available:
        fut = fut.merge(
            stream_out[s]["future"][["feature_date", "predicted_price"]].rename(
                columns={"predicted_price": f"pred_{s}"}
            ),
            on="feature_date",
            how="left",
        )
    pred_cols = [f"pred_{s}" for s in available]
    fut["pred_mean"] = fut[pred_cols].mean(axis=1)
    fut["pred_std"] = fut[pred_cols].std(axis=1).fillna(0.0)
    fut["pred_spread"] = fut[pred_cols].max(axis=1) - fut[pred_cols].min(axis=1)
    feature_cols = pred_cols + ["pred_mean", "pred_std", "pred_spread"] + LSTM_TABULAR_CONTEXT

    run = _latest_meta_run(engine, ticker)
    meta_val_mae = _parse_meta_val_mae(run[1]) if run else None
    meta_path = model_dir(ticker, "xgboost_meta") / "model.json"

    final_future = ensemble_future
    blend_note = f"ensemble-only (ensemble_val_mae={ensemble_val_mae:.4f})"
    if meta_path.exists() and meta_val_mae is not None:
        meta_model = xgb.XGBRegressor()
        meta_model.load_model(meta_path)
        x_fut = fut[feature_cols].astype(float).ffill().bfill()
        meta_future = fut["anchor"].to_numpy(dtype=float) + meta_model.predict(x_fut)
        blend_w = _shrink_weights(
            np.array([1.0 / (ensemble_val_mae + 1e-6), 1.0 / (meta_val_mae + 1e-6)])
        )
        final_future = blend_w[0] * ensemble_future + blend_w[1] * meta_future
        blend_note = (
            f"blend ensemble/meta weights={blend_w.round(3).tolist()} "
            f"(ensemble_val_mae={ensemble_val_mae:.4f}, meta_val_mae={meta_val_mae:.4f})"
        )

    predictions = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(future_dates).dt.date,
            "target_date": pd.to_datetime(
                stream_out[available[0]]["future"]["target_date_30d"]
            ).dt.date,
            "predicted_price": np.round(final_future, 4),
        }
    )
    notes = (
        f"future forecast (reused artifacts); anchor={anchor_mode}; "
        f"streams={available}; ens_weights={ens_weights.round(3).tolist()}; {blend_note}"
    )
    return ForecastResult(ticker, predictions, anchor_mode, notes)


def ensure_future_split_enum(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE predictions "
                "MODIFY split_flag ENUM('train','val','test','future') NOT NULL"
            )
        )


def save_future_predictions(engine: Engine, ticker: str, result: ForecastResult) -> int:
    """Attach future rows to the ticker's latest xgboost_meta run (idempotent)."""
    run = _latest_meta_run(engine, ticker)
    if run is None:
        raise ValueError(f"No xgboost_meta run for {ticker}; train it first.")
    run_id = run[0]

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM predictions WHERE run_id = :run_id AND split_flag = 'future'"
            ),
            {"run_id": run_id},
        )

    out = result.predictions.copy()
    out["run_id"] = run_id
    out["ticker"] = ticker
    out["predicted_at"] = out["feature_date"]
    out["actual_price"] = None
    out["split_flag"] = "future"
    out[
        ["run_id", "ticker", "predicted_at", "target_date",
         "predicted_price", "actual_price", "split_flag"]
    ].to_sql("predictions", engine, if_exists="append", index=False)
    return len(out)
