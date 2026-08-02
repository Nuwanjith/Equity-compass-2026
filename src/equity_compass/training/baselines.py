from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from equity_compass.config import FORECAST_HORIZON
from equity_compass.training.data import (
    EXOG_FEATURES,
    TABULAR_FEATURES,
    get_history_closes,
    get_history_exog,
)

ORDER_CANDIDATES: list[tuple[int, int, int]] = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
]

EXOG_SUBSET_CANDIDATES: list[list[str]] = [
    ["forex_close", "bond_rate", "eps"],
    ["forex_close", "bond_rate"],
    ["forex_close", "eps"],
    ["bond_rate", "eps"],
    ["forex_close"],
    ["bond_rate"],
]

SARIMAX_MAXITER = 200
TUNING_SUBSAMPLE = 4


def predict_naive(df: pd.DataFrame) -> pd.DataFrame:
    """30-day naive forecast: future price equals today's close."""
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    eval_df["predicted_price"] = eval_df["close_price"].astype(float)
    return eval_df


def _fit_exog_scaler(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    train = df[df["split_flag"] == "train"][columns].astype(float).ffill().bfill()
    scaler: dict[str, tuple[float, float]] = {}
    for col in columns:
        mean = float(train[col].mean())
        std = float(train[col].std())
        if std == 0 or np.isnan(std):
            std = 1.0
        scaler[col] = (mean, std)
    return scaler


def _scale_exog_frame(
    exog_df: pd.DataFrame, scaler: dict[str, tuple[float, float]], columns: list[str]
) -> pd.DataFrame:
    out = exog_df[columns].copy().astype(float)
    for col in columns:
        mean, std = scaler[col]
        out[col] = (out[col] - mean) / std
    return out


def _available_exog_columns(df: pd.DataFrame) -> list[str]:
    train = df[df["split_flag"] == "train"]
    available = []
    for col in EXOG_FEATURES:
        if train[col].notna().sum() >= 100:
            available.append(col)
    return available


def _valid_exog_subsets(df: pd.DataFrame) -> list[list[str]]:
    available = set(_available_exog_columns(df))
    subsets = []
    for candidate in EXOG_SUBSET_CANDIDATES:
        cols = [col for col in candidate if col in available]
        if cols and cols not in subsets:
            subsets.append(cols)
    return subsets or [col for col in EXOG_FEATURES if col in available]


def _forecast_exog_matrix(
    exog_hist: pd.DataFrame, horizon: int, columns: list[str]
) -> pd.DataFrame:
    """Forecast each exog series horizon steps ahead with a simple ARIMA(1,1,1)."""
    from statsmodels.tsa.arima.model import ARIMA

    future: dict[str, list[float]] = {}
    for col in columns:
        series = exog_hist[col].astype(float).ffill().bfill()
        if len(series) < 20:
            future[col] = [float(series.iloc[-1])] * horizon
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                future[col] = forecast.astype(float).tolist()
            except Exception:
                future[col] = [float(series.iloc[-1])] * horizon
    return pd.DataFrame(future)


def _arima_forecast(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> float:
    if len(history) < max(order) + 15:
        return float(history.iloc[-1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if exog is not None:
                model = SARIMAX(
                    history,
                    exog=exog,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=SARIMAX_MAXITER)
                if exog_future is not None:
                    forecast = fitted.forecast(steps=horizon, exog=exog_future)
                else:
                    forecast = fitted.forecast(steps=horizon)
            else:
                from statsmodels.tsa.arima.model import ARIMA

                model = ARIMA(history, order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
            return float(forecast.iloc[-1])
        except Exception:
            return float(history.iloc[-1])


def _predict_row_arima(
    df: pd.DataFrame,
    row: pd.Series,
    order: tuple[int, int, int],
    exog_cols: list[str] | None = None,
    exog_scaler: dict[str, tuple[float, float]] | None = None,
) -> float:
    history = get_history_closes(df, row)
    if not exog_cols:
        return _arima_forecast(history, FORECAST_HORIZON, order)

    raw_exog = get_history_exog(df, row, exog_cols).ffill().bfill()
    exog_hist = _scale_exog_frame(raw_exog, exog_scaler, exog_cols)
    exog_future = _forecast_exog_matrix(raw_exog, FORECAST_HORIZON, exog_cols)
    exog_future = _scale_exog_frame(exog_future, exog_scaler, exog_cols)
    return _arima_forecast(
        history,
        FORECAST_HORIZON,
        order,
        exog=exog_hist,
        exog_future=exog_future,
    )


def _score_arima_config(
    df: pd.DataFrame,
    order: tuple[int, int, int],
    split_flag: str,
    exog_cols: list[str] | None = None,
    exog_scaler: dict[str, tuple[float, float]] | None = None,
) -> float:
    subset = df[df["split_flag"] == split_flag].iloc[::TUNING_SUBSAMPLE]
    if subset.empty:
        return float("inf")

    errors = []
    for _, row in subset.iterrows():
        pred = _predict_row_arima(df, row, order, exog_cols, exog_scaler)
        actual = float(row["target_close_30d"])
        errors.append(abs(actual - pred))
    return float(np.mean(errors))


def _tune_arima_order(df: pd.DataFrame) -> tuple[tuple[int, int, int], str]:
    best_order = (1, 1, 1)
    best_mae = float("inf")
    for order in ORDER_CANDIDATES:
        mae = _score_arima_config(df, order, split_flag="val")
        if mae < best_mae:
            best_mae = mae
            best_order = order
    note = f"ARIMA tuned on val (subsample={TUNING_SUBSAMPLE}): order={best_order}, val_mae={best_mae:.4f}"
    return best_order, note


def _tune_arimax_config(
    df: pd.DataFrame,
) -> tuple[tuple[int, int, int], list[str], dict[str, tuple[float, float]], str]:
    subsets = _valid_exog_subsets(df)
    best_order = (1, 1, 1)
    best_cols = subsets[0]
    best_scaler = _fit_exog_scaler(df, best_cols)
    best_mae = float("inf")

    for exog_cols in subsets:
        scaler = _fit_exog_scaler(df, exog_cols)
        for order in ORDER_CANDIDATES:
            mae = _score_arima_config(
                df, order, split_flag="val", exog_cols=exog_cols, exog_scaler=scaler
            )
            if mae < best_mae:
                best_mae = mae
                best_order = order
                best_cols = exog_cols
                best_scaler = scaler

    note = (
        f"ARIMAX tuned on val (subsample={TUNING_SUBSAMPLE}): "
        f"order={best_order}, exog={best_cols}, scaled=True, "
        f"future_exog=ARIMA(1,1,1), val_mae={best_mae:.4f}"
    )
    return best_order, best_cols, best_scaler, note


def predict_arima(df: pd.DataFrame, order: tuple[int, int, int] | None = None) -> pd.DataFrame:
    if order is None:
        order, tune_note = _tune_arima_order(df)
    else:
        tune_note = f"ARIMA fixed order={order}"

    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    preds = []
    for _, row in eval_df.iterrows():
        preds.append(_predict_row_arima(df, row, order))
    eval_df["predicted_price"] = preds
    eval_df.attrs["tuning_notes"] = tune_note
    return eval_df


def predict_arimax(df: pd.DataFrame, order: tuple[int, int, int] | None = None) -> pd.DataFrame:
    if order is None:
        order, exog_cols, exog_scaler, tune_note = _tune_arimax_config(df)
    else:
        exog_cols = _valid_exog_subsets(df)[0]
        exog_scaler = _fit_exog_scaler(df, exog_cols)
        tune_note = f"ARIMAX fixed order={order}, exog={exog_cols}"

    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    preds = []
    for _, row in eval_df.iterrows():
        preds.append(_predict_row_arima(df, row, order, exog_cols, exog_scaler))
    eval_df["predicted_price"] = preds
    eval_df.attrs["tuning_notes"] = tune_note
    return eval_df


def predict_linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["split_flag"] == "train"].copy()
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()

    X_train = train_df[TABULAR_FEATURES].ffill().bfill().astype(float)
    y_train = train_df["target_close_30d"].astype(float)
    X_eval = eval_df[TABULAR_FEATURES].ffill().bfill().astype(float)

    model = LinearRegression()
    model.fit(X_train, y_train)
    eval_df["predicted_price"] = model.predict(X_eval)
    return eval_df


def predict_xgboost_tabular(df: pd.DataFrame) -> pd.DataFrame:
    import xgboost as xgb

    train_df = df[df["split_flag"] == "train"].copy()
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()

    X_train = train_df[TABULAR_FEATURES].ffill().bfill().astype(float)
    y_train = train_df["target_close_30d"].astype(float)
    X_eval = eval_df[TABULAR_FEATURES].ffill().bfill().astype(float)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    eval_df["predicted_price"] = model.predict(X_eval)
    return eval_df


BASELINE_MODELS = {
    "naive": predict_naive,
    "arima": predict_arima,
    "arimax": predict_arimax,
    "linear_regression": predict_linear_regression,
    "xgboost_tabular": predict_xgboost_tabular,
}
