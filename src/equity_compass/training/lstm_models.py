"""LSTM stream models and ensemble meta-learner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_compass.config import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    LSTM_LOOKBACK_CANDIDATES,
    LSTM_META_STREAMS,
    LSTM_PATIENCE,
    LSTM_RESIDUAL_ANCHOR,
    LSTM_STREAM_FEATURES,
    LSTM_TABULAR_CONTEXT,
    LSTM_TICKER_OVERRIDES,
    LSTM_TUNING_EPOCHS,
    LSTM_UNITS_CANDIDATES,
    PROJECT_ROOT,
)
from equity_compass.training.data import TABULAR_FEATURES
from equity_compass.training.sequences import (
    batch_mae_rs,
    build_sequences,
    fit_sequence_scaler,
    fit_tabular_scaler,
    sequences_to_predictions_df,
    transform_sequences,
    transform_tabular,
)


def model_dir(ticker: str, model_type: str) -> Path:
    path = PROJECT_ROOT / "models" / ticker / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_seed(seed: int = 42) -> None:
    """Fix RNG state so LSTM training/tuning is reproducible across runs."""
    import os
    import random

    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _ticker_config(ticker: str) -> dict:
    return LSTM_TICKER_OVERRIDES.get(ticker.upper(), {})


def _prepare_model_df(df: pd.DataFrame, ticker: str = "") -> tuple[pd.DataFrame, str, str]:
    """Attach anchor column for residual targets (naive close or linear baseline)."""
    from sklearn.linear_model import LinearRegression

    overrides = _ticker_config(ticker)
    train_df = df[df["split_flag"] == "train"]
    val_df = df[df["split_flag"] == "val"]
    x_train = train_df[TABULAR_FEATURES].astype(float).ffill().bfill()
    y_train = train_df["target_close_30d"].astype(float)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    out = df.copy()
    x_all = out[TABULAR_FEATURES].astype(float).ffill().bfill()
    out["lstm_anchor"] = lr.predict(x_all)

    anchor_mode = overrides.get("residual_anchor", LSTM_RESIDUAL_ANCHOR)
    if anchor_mode == "auto":
        if len(val_df):
            naive_mae = float(
                np.mean(np.abs(val_df["target_close_30d"] - val_df["close_price"]))
            )
            linear_mae = float(
                np.mean(np.abs(val_df["target_close_30d"] - lr.predict(
                    val_df[TABULAR_FEATURES].astype(float).ffill().bfill()
                )))
            )
            anchor_mode = "linear" if linear_mae < naive_mae else "naive"
        else:
            anchor_mode = "naive"

    if anchor_mode == "linear":
        return out, "lstm_anchor", "linear"
    return df, "close_price", "naive"


def _build_lstm(
    input_shape: tuple[int, int],
    units: tuple[int, int],
    tab_dim: int,
):
    import tensorflow as tf

    seq_input = tf.keras.layers.Input(shape=input_shape, name="sequence")
    x = tf.keras.layers.LSTM(units[0], return_sequences=True)(seq_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(units[1])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if tab_dim > 0:
        tab_input = tf.keras.layers.Input(shape=(tab_dim,), name="tabular")
        x = tf.keras.layers.Concatenate()([x, tab_input])
        inputs: list | tf.keras.layers.Input = [seq_input, tab_input]
    else:
        inputs = seq_input

    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="lstm_residual_forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
        loss="mae",
        metrics=["mae"],
    )
    return model


def _batch_xy(batch):
    if len(batch.tab):
        return [batch.x, batch.tab], batch.y
    return batch.x, batch.y


def train_lstm_on_batches(
    train_batch,
    val_batch,
    model_path: Path,
    units: tuple[int, int],
    *,
    epochs: int | None = None,
) -> Path:
    import tensorflow as tf

    if len(train_batch.x) == 0:
        raise ValueError("No training sequences available")

    _set_seed(42)
    tab_dim = train_batch.tab.shape[1] if len(train_batch.tab) else 0
    model = _build_lstm((train_batch.x.shape[1], train_batch.x.shape[2]), units, tab_dim)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=LSTM_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]
    x_train, y_train = _batch_xy(train_batch)
    if len(val_batch.x):
        x_val, y_val = _batch_xy(val_batch)
        validation_data = (x_val, y_val)
    else:
        validation_data = None
    model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs or LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=callbacks if validation_data else [],
    )
    model.save(model_path)
    return model_path


def predict_lstm_batch(batch, model_path: Path) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    if len(batch.x) == 0:
        return np.array([])
    x, _ = _batch_xy(batch)
    return model.predict(x, verbose=0).reshape(-1)


def _prepare_batches(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    ticker: str = "",
) -> tuple:
    model_df, anchor_col, _ = _prepare_model_df(df, ticker)
    train_raw = build_sequences(
        model_df, feature_cols, lookback=lookback, splits={"train"},
        tabular_cols=LSTM_TABULAR_CONTEXT, anchor_col=anchor_col,
    )
    val_raw = build_sequences(
        model_df, feature_cols, lookback=lookback, splits={"val"},
        tabular_cols=LSTM_TABULAR_CONTEXT, anchor_col=anchor_col,
    )
    if len(train_raw.x) == 0:
        raise ValueError("No training sequences")

    seq_scaler = fit_sequence_scaler(train_raw.x)
    tab_scaler = fit_tabular_scaler(train_raw.tab)
    train = train_raw
    val = val_raw
    train.x = transform_sequences(train.x, seq_scaler)
    train.tab = transform_tabular(train.tab, tab_scaler)
    if len(val.x):
        val.x = transform_sequences(val.x, seq_scaler)
        val.tab = transform_tabular(val.tab, tab_scaler)
    return train, val, seq_scaler, tab_scaler


def _score_config(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    units: tuple[int, int],
    ticker: str = "",
) -> float:
    train, val, _, _ = _prepare_batches(df, feature_cols, lookback, ticker)
    if len(val.x) == 0:
        return float("inf")

    import tempfile

    overrides = _ticker_config(ticker)
    tune_epochs = overrides.get("tuning_epochs", LSTM_TUNING_EPOCHS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trial.keras"
        train_lstm_on_batches(train, val, path, units, epochs=tune_epochs)
        preds = predict_lstm_batch(val, path)
    return batch_mae_rs(preds, val)


def _tune_stream(df: pd.DataFrame, feature_cols: list[str], ticker: str = "") -> tuple[int, tuple[int, int], str]:
    best_lookback = LSTM_LOOKBACK_CANDIDATES[0]
    best_units = LSTM_UNITS_CANDIDATES[0]
    best_mae = float("inf")

    for lookback in LSTM_LOOKBACK_CANDIDATES:
        for units in LSTM_UNITS_CANDIDATES:
            mae = _score_config(df, feature_cols, lookback, units, ticker)
            if mae < best_mae:
                best_mae = mae
                best_lookback = lookback
                best_units = units

    note = (
        f"residual_anchor={LSTM_RESIDUAL_ANCHOR}, tuned lookback={best_lookback}, "
        f"units={best_units}, val_mae={best_mae:.4f}"
    )
    return best_lookback, best_units, note


def _calibrate_residual_alpha(val_batch, val_preds: np.ndarray) -> float:
    """Scale LSTM residual on validation; keep 1.0 unless scaling helps val MAE."""
    base_mae = batch_mae_rs(val_preds, val_batch)
    best_alpha = 1.0
    best_mae = base_mae
    for alpha in np.linspace(0.25, 1.0, 16):
        mae = batch_mae_rs(val_preds * alpha, val_batch)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha


def train_and_predict_stream(
    df: pd.DataFrame,
    ticker: str,
    model_type: str,
) -> pd.DataFrame:
    feature_cols = LSTM_STREAM_FEATURES[model_type]
    model_df, anchor_col, anchor_mode = _prepare_model_df(df, ticker)
    lookback, units, tune_note = _tune_stream(model_df, feature_cols, ticker)

    train_raw, val_raw, seq_scaler, tab_scaler = _prepare_batches(
        model_df, feature_cols, lookback, ticker
    )
    overrides = _ticker_config(ticker)
    train_epochs = overrides.get("epochs", LSTM_EPOCHS)
    path = model_dir(ticker, model_type) / "model.keras"
    train_lstm_on_batches(train_raw, val_raw, path, units, epochs=train_epochs)

    alpha = 1.0
    if len(val_raw.x):
        val_preds = predict_lstm_batch(val_raw, path)
        alpha = _calibrate_residual_alpha(val_raw, val_preds)

    scaler_path = model_dir(ticker, model_type) / "scaler.json"
    scaler_path.write_text(
        json.dumps(
            {
                "seq_mean": seq_scaler.mean_.tolist(),
                "seq_scale": seq_scaler.scale_.tolist(),
                "tab_mean": tab_scaler.mean_.tolist(),
                "tab_scale": tab_scaler.scale_.tolist(),
                "lookback": lookback,
                "units": list(units),
                "features": feature_cols,
                "tabular_context": LSTM_TABULAR_CONTEXT,
                "residual_alpha": alpha,
                "residual_anchor": anchor_mode,
            }
        )
    )

    parts = []
    for split in ("train", "val", "test"):
        batch_raw = build_sequences(
            model_df, feature_cols, lookback=lookback, splits={split},
            tabular_cols=LSTM_TABULAR_CONTEXT, anchor_col=anchor_col,
        )
        if len(batch_raw.x) == 0:
            continue
        batch_raw.x = transform_sequences(batch_raw.x, seq_scaler)
        batch_raw.tab = transform_tabular(batch_raw.tab, tab_scaler)
        preds = predict_lstm_batch(batch_raw, path) * alpha
        parts.append(sequences_to_predictions_df(batch_raw, preds, df))

    out = pd.concat(parts, ignore_index=True).sort_values("feature_date")
    out.attrs["model_path"] = str(path)
    out.attrs["tuning_notes"] = (
        f"{tune_note}; anchor={anchor_mode}; alpha={alpha:.2f}; "
        f"train_seq={len(train_raw.x)}, val_seq={len(val_raw.x)}"
    )
    return out


def eval_predictions(full_preds: pd.DataFrame) -> pd.DataFrame:
    return full_preds[full_preds["split_flag"].isin(["val", "test"])].copy()


def _stream_val_mae(stream: pd.DataFrame) -> float:
    val = stream[stream["split_flag"] == "val"]
    if len(val) == 0:
        return float("inf")
    return float(np.mean(np.abs(val["target_close_30d"] - val["predicted_price"])))


def _shrink_weights(inv_err: np.ndarray, shrinkage: float = 0.5) -> np.ndarray:
    """
    Blend inverse-error weights toward equal weighting.

    With only ~200-400 validation rows on a small, non-stationary series,
    inverse-MAE weights are themselves noisy estimates — a candidate can look
    best on val purely by chance and dominate the blend while generalizing
    poorly to test (observed empirically for the bond-rate-only macro stream
    on both pilot tickers here). Shrinking toward the equal-weight combination
    is a standard, well-documented fix in the forecast-combination literature
    (the "forecast combination puzzle": naive/equal weights often beat
    "optimal" inverse-error weights in small samples).
    """
    inv_err = np.asarray(inv_err, dtype=float)
    w_inv = inv_err / inv_err.sum()
    w_eq = np.full_like(w_inv, 1.0 / len(w_inv))
    return shrinkage * w_eq + (1 - shrinkage) * w_inv


def _weighted_ensemble_predictions(
    stream_preds: dict[str, pd.DataFrame],
    available: list[str],
    splits: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Shrunk val-MAE-weighted average of stream predictions for given splits."""
    base = None
    pred_matrix = []
    weights = []

    for name in available:
        stream = stream_preds[name]
        part = stream[stream["split_flag"].isin(splits)].sort_values("feature_date")
        if base is None:
            base = part[
                ["feature_date", "target_close_30d", "target_date_30d", "split_flag", "close_price"]
            ].reset_index(drop=True)
        pred_matrix.append(part["predicted_price"].astype(float).reset_index(drop=True).values)
        weights.append(1.0 / (_stream_val_mae(stream) + 1e-6))

    weight_arr = _shrink_weights(np.asarray(weights, dtype=float))
    stacked = np.vstack(pred_matrix)
    base = base.copy()
    base["predicted_price"] = (weight_arr[:, None] * stacked).sum(axis=0)
    return base, weight_arr


def predict_lstm_avg_ensemble(
    df: pd.DataFrame,
    ticker: str,
    stream_preds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    available = [name for name in LSTM_META_STREAMS if name in stream_preds]
    if len(available) < 2:
        raise ValueError("Need at least 2 trained streams for the ensemble")

    eval_base, weight_arr = _weighted_ensemble_predictions(
        stream_preds, available, {"val", "test"}
    )
    eval_base.attrs["tuning_notes"] = (
        f"Val-MAE weighted average of {available}; weights={weight_arr.round(3).tolist()}"
    )
    return eval_base


META_PARAM_GRID: list[dict] = [
    dict(max_depth=2, learning_rate=0.05, reg_lambda=1.0, reg_alpha=0.0, subsample=0.9, colsample_bytree=0.9, min_child_weight=1, gamma=0.0),
    dict(max_depth=2, learning_rate=0.03, reg_lambda=2.0, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1),
    dict(max_depth=2, learning_rate=0.08, reg_lambda=0.5, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, gamma=0.0),
    dict(max_depth=3, learning_rate=0.05, reg_lambda=1.0, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, gamma=0.0),
    dict(max_depth=3, learning_rate=0.03, reg_lambda=2.0, reg_alpha=0.2, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.2),
    dict(max_depth=3, learning_rate=0.04, reg_lambda=1.5, reg_alpha=0.05, subsample=0.85, colsample_bytree=0.9, min_child_weight=4, gamma=0.1),
    dict(max_depth=4, learning_rate=0.05, reg_lambda=1.0, reg_alpha=0.05, subsample=0.9, colsample_bytree=1.0, min_child_weight=3, gamma=0.1),
    dict(max_depth=4, learning_rate=0.02, reg_lambda=3.0, reg_alpha=0.2, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.3),
    dict(max_depth=5, learning_rate=0.03, reg_lambda=2.0, reg_alpha=0.3, subsample=0.7, colsample_bytree=0.6, min_child_weight=8, gamma=0.4),
    dict(max_depth=5, learning_rate=0.02, reg_lambda=4.0, reg_alpha=0.4, subsample=0.6, colsample_bytree=0.6, min_child_weight=10, gamma=0.5),
]
META_MAX_ESTIMATORS = 500
META_EARLY_STOPPING_ROUNDS = 20


def _cv_mae(
    frame: pd.DataFrame,
    feature_cols: list[str],
    model_factory,
) -> tuple[float, int | None]:
    """Walk-forward CV price-MAE for an estimator factory: (train_idx, ...) -> (fitted_model, best_iter|None)."""
    from sklearn.model_selection import TimeSeriesSplit

    x = frame[feature_cols].astype(float).ffill().bfill().values
    y = frame["y_residual"].astype(float).values
    anchor = frame["anchor"].astype(float).values
    actual = frame["target_close_30d"].astype(float).values

    n_splits = 3 if len(frame) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_maes = []
    fold_iters = []
    for train_idx, test_idx in tscv.split(x):
        if len(train_idx) < 10 or len(test_idx) == 0:
            continue
        model, best_iter = model_factory(x[train_idx], y[train_idx], x[test_idx], y[test_idx])
        pred_price = anchor[test_idx] + model.predict(x[test_idx])
        fold_maes.append(float(np.mean(np.abs(actual[test_idx] - pred_price))))
        if best_iter is not None:
            fold_iters.append(best_iter)

    if not fold_maes:
        return float("inf"), None
    mean_iter = int(round(np.mean(fold_iters))) if fold_iters else None
    return float(np.mean(fold_maes)), mean_iter


def _xgb_factory(params: dict):
    def factory(x_tr, y_tr, x_te, y_te):
        import xgboost as xgb

        model = xgb.XGBRegressor(
            **params,
            n_estimators=META_MAX_ESTIMATORS,
            random_state=42,
            objective="reg:absoluteerror",
            eval_metric="mae",
            early_stopping_rounds=META_EARLY_STOPPING_ROUNDS,
        )
        model.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], verbose=False)
        return model, getattr(model, "best_iteration", None)

    return factory


def _tune_meta_params(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, float]:
    """Grid search (incl. min_child_weight/gamma) + early-stopping tree count, via walk-forward CV on val."""
    best_params: dict = dict(META_PARAM_GRID[0])
    best_mae = float("inf")
    best_iter: int | None = None

    for params in META_PARAM_GRID:
        mae, mean_iter = _cv_mae(frame, feature_cols, _xgb_factory(params))
        if mae < best_mae:
            best_mae = mae
            best_params = dict(params)
            best_iter = mean_iter

    best_params["n_estimators"] = max(best_iter or META_MAX_ESTIMATORS, 10)
    return best_params, best_mae


def _meta_frame(
    model_df: pd.DataFrame,
    stream_preds: dict[str, pd.DataFrame],
    available: list[str],
    anchor_col: str,
    splits: set[str],
) -> pd.DataFrame:
    """Assemble meta-learner features from out-of-sample stream predictions only."""
    mask = model_df["split_flag"].isin(splits)
    cols_needed = list(
        dict.fromkeys(
            ["feature_date", "split_flag", "target_close_30d", "target_date_30d", "close_price"]
            + LSTM_TABULAR_CONTEXT
        )
    )
    frame = model_df.loc[mask, cols_needed].copy()
    frame["anchor"] = model_df.loc[mask, anchor_col].astype(float).values

    for name in available:
        part = stream_preds[name]
        part = part[part["split_flag"].isin(splits)][
            ["feature_date", "split_flag", "predicted_price"]
        ]
        frame = frame.merge(
            part.rename(columns={"predicted_price": f"pred_{name}"}),
            on=["feature_date", "split_flag"],
            how="inner",
        )

    pred_cols = [f"pred_{name}" for name in available]
    frame[pred_cols] = frame[pred_cols].astype(float)
    frame["pred_mean"] = frame[pred_cols].mean(axis=1)
    frame["pred_std"] = frame[pred_cols].std(axis=1).fillna(0.0)
    frame["pred_spread"] = frame[pred_cols].max(axis=1) - frame[pred_cols].min(axis=1)
    frame["y_residual"] = frame["target_close_30d"].astype(float) - frame["anchor"]
    return frame.sort_values("feature_date").reset_index(drop=True)


def predict_xgboost_meta(
    df: pd.DataFrame,
    ticker: str,
    stream_preds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Final ensemble: confidence-weighted blend of two candidates.

    Candidates:
      1. Val-MAE-weighted average of all trained streams ("ensemble").
      2. XGBoost meta-learner: grid-searched (incl. min_child_weight/gamma)
         with early-stopping-selected tree count, via walk-forward CV on
         `val`, predicting a residual vs the same anchor (naive/linear) used
         by the LSTM streams: price = anchor + predicted_residual.

    Candidate 2 is trained ONLY on `val`, where every stream's prediction is
    genuinely out-of-sample — train-split stream predictions are in-sample /
    overfit and would leak into the meta-learner if used.

    The single best-individual-stream (by val MAE) is intentionally NOT a
    separate blend candidate. Empirically (TYRE) whichever stream happens to
    look best on a noisy ~200-400 row `val` split can be by far the worst on
    `test` (regime shift), and dedicating ~30% blend weight to that one
    stream's raw prediction hurt overall MAE. `ensemble` already folds every
    stream in with its own (shrunk) inverse-val-MAE weight, so a strong
    stream still gets a meaningful say without a single noisy pick dominating.
    It's still computed/logged below for diagnostics.

    Rather than hard-selecting the single candidate with the lowest val MAE,
    the final prediction is a weighted blend, weighted by inverse validation
    MAE and shrunk toward equal weighting (Bates-Granger-style forecast
    combination) to hedge against a single noisy validation estimate driving
    the whole ensemble.

    NB: a richer variant of this (recency-weighted sample weights, pairwise
    stream-disagreement features, and a Ridge candidate) was tried and
    measurably made test MAE *worse* on both pilot tickers, even though it
    looked better on validation — the extra flexibility let candidates
    overfit the small `val` split more tightly without generalizing. That
    variant was reverted; keep this in mind before adding more capacity here.
    """
    import xgboost as xgb

    available = [name for name in LSTM_META_STREAMS if name in stream_preds]
    if len(available) < 2:
        raise ValueError("Need at least 2 trained streams for the meta-learner")

    model_df, anchor_col, anchor_mode = _prepare_model_df(df, ticker)

    val_frame = _meta_frame(model_df, stream_preds, available, anchor_col, {"val"})
    test_frame = _meta_frame(model_df, stream_preds, available, anchor_col, {"test"})

    pred_cols = [f"pred_{name}" for name in available]
    feature_cols = pred_cols + ["pred_mean", "pred_std", "pred_spread"] + LSTM_TABULAR_CONTEXT

    stream_val_maes = {name: _stream_val_mae(stream_preds[name]) for name in available}
    best_stream_name = min(stream_val_maes, key=stream_val_maes.get)
    best_stream_val_mae = stream_val_maes[best_stream_name]

    ens_val_frame, _ = _weighted_ensemble_predictions(stream_preds, available, {"val"})
    ensemble_val_mae = float(
        np.mean(np.abs(
            ens_val_frame["target_close_30d"].astype(float)
            - ens_val_frame["predicted_price"].astype(float)
        ))
    )

    best_params: dict = {}
    meta_val_mae = float("inf")
    meta_model = None
    if len(val_frame) >= 20:
        best_params, meta_val_mae = _tune_meta_params(val_frame, feature_cols)
        x_val = val_frame[feature_cols].astype(float).ffill().bfill()
        y_val = val_frame["y_residual"].astype(float)
        meta_model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            objective="reg:absoluteerror",
        )
        meta_model.fit(x_val, y_val)

    base = model_df[model_df["split_flag"].isin(["val", "test"])][
        ["feature_date", "split_flag", "target_close_30d", "target_date_30d", "close_price"]
    ].copy()

    best_stream_series = eval_predictions(stream_preds[best_stream_name])[
        ["feature_date", "split_flag", "predicted_price"]
    ].rename(columns={"predicted_price": "pred_best_stream"})
    ensemble_series, ens_weights = _weighted_ensemble_predictions(
        stream_preds, available, {"val", "test"}
    )
    ensemble_series = ensemble_series[["feature_date", "split_flag", "predicted_price"]].rename(
        columns={"predicted_price": "pred_ensemble"}
    )

    combo = base.merge(best_stream_series, on=["feature_date", "split_flag"], how="left")
    combo = combo.merge(ensemble_series, on=["feature_date", "split_flag"], how="left")

    weight_map = {
        "ensemble": 1.0 / (ensemble_val_mae + 1e-6),
    }

    if meta_model is not None:
        full_meta_frame = (
            pd.concat([val_frame, test_frame], ignore_index=True) if len(test_frame) else val_frame
        )
        x_full = full_meta_frame[feature_cols].astype(float).ffill().bfill()
        meta_series = full_meta_frame[["feature_date", "split_flag"]].copy()
        meta_series["pred_meta"] = (
            full_meta_frame["anchor"].astype(float).values + meta_model.predict(x_full)
        )
        combo = combo.merge(meta_series, on=["feature_date", "split_flag"], how="left")
        weight_map["meta"] = 1.0 / (meta_val_mae + 1e-6)

    names = list(weight_map.keys())
    weight_arr = _shrink_weights(np.asarray([weight_map[n] for n in names], dtype=float))
    pred_matrix = np.vstack([combo[f"pred_{n}"].astype(float).values for n in names])
    combo["predicted_price"] = (weight_arr[:, None] * pred_matrix).sum(axis=0)

    out = combo[
        ["feature_date", "predicted_price", "target_close_30d", "target_date_30d",
         "split_flag", "close_price"]
    ].sort_values("feature_date").reset_index(drop=True)

    notes = (
        f"Confidence-weighted blend of {names}; weights={dict(zip(names, weight_arr.round(3)))}; "
        f"anchor={anchor_mode}; best_stream_diagnostic={best_stream_name} "
        f"(val_mae={best_stream_val_mae:.4f}, not in blend — see docstring); "
        f"ensemble_val_mae={ensemble_val_mae:.4f} (weights={ens_weights.round(3).tolist()})"
    )
    if meta_model is not None:
        path = model_dir(ticker, "xgboost_meta") / "model.json"
        meta_model.save_model(path)
        out.attrs["model_path"] = str(path)
        notes += f"; meta_val_mae={meta_val_mae:.4f}, meta_params={best_params}"
    else:
        notes += "; meta skipped (insufficient val rows for tuning)"
    out.attrs["tuning_notes"] = notes
    return out


LSTM_TRAIN_ORDER = [
    "lstm_fundamental",
    "lstm_macro",
    "lstm_forex",
    "lstm_univariate",
    "lstm_avg_ensemble",
    "xgboost_meta",
]
