"""Build LSTM sequence tensors from tabular features_daily rows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from equity_compass.config import LOOKBACK


@dataclass
class SequenceBatch:
    """Sequence input, optional tabular context at day t, residual targets."""

    x: np.ndarray
    y: np.ndarray
    anchor_close: np.ndarray
    y_actual: np.ndarray
    feature_dates: np.ndarray
    target_dates: np.ndarray
    split_flags: np.ndarray
    tab: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))


def _allowed_window_splits(end_split: str) -> set[str]:
    if end_split == "train":
        return {"train"}
    if end_split == "val":
        return {"train", "val"}
    if end_split == "test":
        return {"train", "val", "test"}
    return set()


def fit_sequence_scaler(train_x: np.ndarray) -> StandardScaler:
    """Fit per-feature standardizer on all timesteps in training windows."""
    n_samples, lookback, n_features = train_x.shape
    flat = train_x.reshape(n_samples * lookback, n_features)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def fit_tabular_scaler(train_tab: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    if len(train_tab):
        scaler.fit(train_tab)
    return scaler


def transform_sequences(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if len(x) == 0:
        return x
    n_samples, lookback, n_features = x.shape
    flat = scaler.transform(x.reshape(n_samples * lookback, n_features))
    return flat.reshape(n_samples, lookback, n_features).astype(np.float32)


def transform_tabular(tab: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if len(tab) == 0:
        return tab
    return scaler.transform(tab).astype(np.float32)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    lookback: int = LOOKBACK,
    splits: set[str] | None = None,
    tabular_cols: list[str] | None = None,
    anchor_col: str = "close_price",
) -> SequenceBatch:
    """
    Sliding windows for LSTM training/inference.

    Input window: rows [t - lookback + 1, ..., t]
    Target: price residual = target_close_30d - close_price at day t
    (predicted price = anchor_close + predicted_residual)
    """
    tab_cols = tabular_cols or []
    all_cols = list(dict.fromkeys(feature_cols + tab_cols))

    data = df.copy()
    data[all_cols] = data[all_cols].astype(float).ffill().bfill()
    values = data[feature_cols].to_numpy(dtype=np.float32)
    tab_values = data[tab_cols].to_numpy(dtype=np.float32) if tab_cols else None
    targets = data["target_close_30d"].astype(float).to_numpy(dtype=np.float32)
    anchors = data[anchor_col].astype(float).to_numpy(dtype=np.float32)
    dates = data["feature_date"].to_numpy()
    target_dates = data["target_date_30d"].to_numpy()
    split_flags = data["split_flag"].to_numpy()

    xs: list[np.ndarray] = []
    tabs: list[np.ndarray] = []
    ys: list[float] = []
    ac: list[float] = []
    ya: list[float] = []
    fd: list = []
    td: list = []
    sf: list[str] = []

    for i in range(lookback - 1, len(data)):
        end_split = split_flags[i]
        if splits is not None and end_split not in splits:
            continue

        window_splits = split_flags[i - lookback + 1 : i + 1]
        allowed = _allowed_window_splits(end_split)
        if not set(window_splits).issubset(allowed):
            continue

        xs.append(values[i - lookback + 1 : i + 1])
        if tab_values is not None:
            tabs.append(tab_values[i])
        anchor = float(anchors[i])
        actual = float(targets[i])
        ys.append(actual - anchor)
        ac.append(anchor)
        ya.append(actual)
        fd.append(dates[i])
        td.append(target_dates[i])
        sf.append(end_split)

    empty_tab = np.empty((0, len(tab_cols)), dtype=np.float32)
    if not xs:
        return SequenceBatch(
            x=np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            anchor_close=np.empty((0,), dtype=np.float32),
            y_actual=np.empty((0,), dtype=np.float32),
            feature_dates=np.array([]),
            target_dates=np.array([]),
            split_flags=np.array([]),
            tab=empty_tab,
        )

    return SequenceBatch(
        x=np.stack(xs),
        y=np.asarray(ys, dtype=np.float32),
        anchor_close=np.asarray(ac, dtype=np.float32),
        y_actual=np.asarray(ya, dtype=np.float32),
        feature_dates=np.asarray(fd),
        target_dates=np.asarray(td),
        split_flags=np.asarray(sf),
        tab=np.stack(tabs).astype(np.float32) if tabs else empty_tab,
    )


def residuals_to_prices(residuals: np.ndarray, anchor_close: np.ndarray) -> np.ndarray:
    return residuals.astype(float) + anchor_close.astype(float)


def batch_mae_rs(pred_residuals: np.ndarray, batch: SequenceBatch) -> float:
    if len(batch.y) == 0:
        return float("inf")
    pred_price = residuals_to_prices(pred_residuals, batch.anchor_close)
    return float(np.mean(np.abs(batch.y_actual - pred_price)))


def sequences_to_predictions_df(
    batch: SequenceBatch,
    predicted: np.ndarray,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    pred_price = residuals_to_prices(predicted, batch.anchor_close)
    pred_df = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(batch.feature_dates),
            "predicted_price": pred_price.astype(float),
            "target_close_30d": batch.y_actual.astype(float),
            "target_date_30d": pd.to_datetime(batch.target_dates),
            "split_flag": batch.split_flags,
        }
    )
    merge_cols = ["feature_date", "close_price", "split_flag"]
    return pred_df.merge(
        full_df[merge_cols],
        on=["feature_date", "split_flag"],
        how="left",
    )
