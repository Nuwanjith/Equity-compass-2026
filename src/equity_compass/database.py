from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from equity_compass.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)


def get_engine() -> Engine:
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(url)


def delete_ticker_rows(engine: Engine, ticker: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM features_daily WHERE ticker = :ticker"),
            {"ticker": ticker},
        )
        conn.execute(
            text("DELETE FROM scaler_params WHERE ticker = :ticker"),
            {"ticker": ticker},
        )
