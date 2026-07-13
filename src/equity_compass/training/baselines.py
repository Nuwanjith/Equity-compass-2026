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


def predict_naive(df: pd.DataFrame) -> pd.DataFrame:
    """30-day naive forecast: future price equals today's close."""
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    eval_df["predicted_price"] = eval_df["close_price"].astype(float)
    return eval_df


def _arima_forecast(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
    exog_future: pd.Series | None = None,
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
                fitted = model.fit(disp=False, maxiter=50)
                if exog_future is not None:
                    future_exog = pd.DataFrame(
                        [exog_future.values] * horizon,
                        columns=EXOG_FEATURES,
                    )
                    forecast = fitted.forecast(steps=horizon, exog=future_exog)
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


def predict_arima(
    df: pd.DataFrame, order: tuple[int, int, int] = (1, 1, 1)
) -> pd.DataFrame:
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    preds = []
    for _, row in eval_df.iterrows():
        history = get_history_closes(df, row)
        preds.append(_arima_forecast(history, FORECAST_HORIZON, order))
    eval_df["predicted_price"] = preds
    return eval_df


def predict_arimax(
    df: pd.DataFrame, order: tuple[int, int, int] = (1, 1, 1)
) -> pd.DataFrame:
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()
    preds = []
    for _, row in eval_df.iterrows():
        history = get_history_closes(df, row)
        exog_hist = get_history_exog(df, row).ffill().bfill()
        exog_future = row[EXOG_FEATURES].astype(float).ffill()
        preds.append(
            _arima_forecast(
                history,
                FORECAST_HORIZON,
                order,
                exog=exog_hist,
                exog_future=exog_future,
            )
        )
    eval_df["predicted_price"] = preds
    return eval_df


def predict_linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["split_flag"] == "train"].copy()
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()

    X_train = train_df[TABULAR_FEATURES].ffill().bfill()
    y_train = train_df["target_close_30d"].astype(float)
    X_eval = eval_df[TABULAR_FEATURES].ffill().bfill()

    model = LinearRegression()
    model.fit(X_train, y_train)
    eval_df["predicted_price"] = model.predict(X_eval)
    return eval_df


def predict_xgboost_tabular(df: pd.DataFrame) -> pd.DataFrame:
    import xgboost as xgb

    train_df = df[df["split_flag"] == "train"].copy()
    eval_df = df[df["split_flag"].isin(["val", "test"])].copy()

    X_train = train_df[TABULAR_FEATURES].ffill().bfill()
    y_train = train_df["target_close_30d"].astype(float)
    X_eval = eval_df[TABULAR_FEATURES].ffill().bfill()

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
