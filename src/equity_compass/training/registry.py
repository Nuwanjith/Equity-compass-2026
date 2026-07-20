from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from equity_compass.training.metrics import compute_metrics


def ensure_model_tables(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        model_type ENUM(
            'naive','arima','arimax','linear_regression','xgboost_tabular',
            'lstm_univariate','lstm_fundamental','lstm_macro','lstm_forex',
            'lstm_avg_ensemble','xgboost_meta'
        ) NOT NULL,
        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        train_from DATE NULL, train_to DATE NULL,
        val_from DATE NULL, val_to DATE NULL,
        test_from DATE NULL, test_to DATE NULL,
        mae FLOAT NULL, rmse FLOAT NULL, mape FLOAT NULL,
        r2 FLOAT NULL, directional_accuracy FLOAT NULL,
        model_path VARCHAR(500) NULL, notes TEXT NULL,
        INDEX idx_ticker_model (ticker, model_type)
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id INT NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        predicted_at DATE NOT NULL,
        target_date DATE NOT NULL,
        predicted_price DECIMAL(12,4) NOT NULL,
        actual_price DECIMAL(12,4) NULL,
        split_flag ENUM('train','val','test') NOT NULL,
        FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
        INDEX idx_ticker_target (ticker, target_date),
        INDEX idx_run_id (run_id)
    );
    """
    with engine.begin() as conn:
        for statement in ddl.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))


def delete_model_run(engine: Engine, ticker: str, model_type: str) -> None:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT run_id FROM model_runs "
                "WHERE ticker = :ticker AND model_type = :model_type"
            ),
            {"ticker": ticker, "model_type": model_type},
        ).fetchall()
        for (run_id,) in rows:
            conn.execute(
                text("DELETE FROM predictions WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
        conn.execute(
            text(
                "DELETE FROM model_runs "
                "WHERE ticker = :ticker AND model_type = :model_type"
            ),
            {"ticker": ticker, "model_type": model_type},
        )


def save_run(
    engine: Engine,
    ticker: str,
    model_type: str,
    predictions_df: pd.DataFrame,
    ranges: dict,
    metrics: dict[str, float],
    notes: str | None = None,
    *,
    replace: bool = True,
) -> int:
    if replace:
        delete_model_run(engine, ticker, model_type)

    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO model_runs (
                    ticker, model_type,
                    train_from, train_to, val_from, val_to, test_from, test_to,
                    mae, rmse, mape, r2, directional_accuracy, notes
                ) VALUES (
                    :ticker, :model_type,
                    :train_from, :train_to, :val_from, :val_to,
                    :test_from, :test_to,
                    :mae, :rmse, :mape, :r2, :directional_accuracy, :notes
                )
                """
            ),
            {
                "ticker": ticker,
                "model_type": model_type,
                "train_from": ranges["train"][0],
                "train_to": ranges["train"][1],
                "val_from": ranges["val"][0],
                "val_to": ranges["val"][1],
                "test_from": ranges["test"][0],
                "test_to": ranges["test"][1],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "directional_accuracy": metrics["directional_accuracy"],
                "notes": notes,
            },
        )
        run_id = result.lastrowid

    pred_out = predictions_df[
        ["feature_date", "target_date_30d", "predicted_price", "target_close_30d", "split_flag"]
    ].copy()
    pred_out = pred_out.rename(
        columns={
            "feature_date": "predicted_at",
            "target_date_30d": "target_date",
            "target_close_30d": "actual_price",
        }
    )
    pred_out["run_id"] = run_id
    pred_out["ticker"] = ticker
    pred_out["target_date"] = pd.to_datetime(pred_out["target_date"]).dt.date

    pred_out[
        ["run_id", "ticker", "predicted_at", "target_date",
         "predicted_price", "actual_price", "split_flag"]
    ].to_sql("predictions", engine, if_exists="append", index=False)

    return run_id


def evaluate_and_save(
    engine: Engine,
    ticker: str,
    model_type: str,
    predictions_df: pd.DataFrame,
    ranges: dict,
    *,
    eval_split: str = "test",
    replace: bool = True,
    notes: str | None = None,
) -> dict:
    eval_df = predictions_df[predictions_df["split_flag"] == eval_split]
    y_true = eval_df["target_close_30d"].astype(float).values
    y_pred = eval_df["predicted_price"].astype(float).values
    metrics = compute_metrics(y_true, y_pred)

    if notes is None:
        notes = f"Evaluated on {eval_split} split ({len(eval_df)} rows)"

    save_run(
        engine, ticker, model_type, predictions_df, ranges, metrics,
        notes=notes,
        replace=replace,
    )
    return metrics
