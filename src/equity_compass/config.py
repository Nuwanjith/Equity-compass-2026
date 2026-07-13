from pathlib import Path

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "config" / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root53421")
DB_NAME = os.getenv("DB_NAME", "equity_compass")

FORECAST_HORIZON = 30
BOND_START = "2014-09-02"
PIPELINE_VERSION = "v1"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TICKERS = [
    "CARG", "COMB", "CTC", "DIPD", "HAYC", "HAYL",
    "KCAB", "KVAL", "MELS", "SAMP", "TYRE",
]

FEATURES_TO_SCALE = {
    "close_price": "close_scaled",
    "eps": "eps_scaled",
    "bond_rate": "bond_rate_scaled",
    "forex_close": "forex_close_scaled",
    "target_close_30d": "target_close_30d_scaled",
}
