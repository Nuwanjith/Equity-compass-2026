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
PIPELINE_VERSION = "v2"

# Back-adjust pre-split prices/fundamentals to post-split basis during preprocessing.
# DIPD: 10-for-1 subdivision; first regular post-split session ~2021-02-16.
CORPORATE_ACTIONS: dict[str, dict[str, dict[str, float | str]]] = {
    "DIPD": {
        "subdivision": {
            "effective_date": "2021-02-16",
            "ratio": 10.0,
            "transition_through": "2021-02-15",
            "transition_price_threshold": 100.0,
        }
    }
}

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

# Phase 5 — LSTM sequence models (residual target, tuned on val MAE)
LOOKBACK = 60
LSTM_LOOKBACK_CANDIDATES = [30, 45, 60, 90]
LSTM_UNITS_CANDIDATES = [(32, 16), (64, 32), (128, 64)]
LSTM_UNITS = (64, 32)
LSTM_EPOCHS = 120
LSTM_TUNING_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 12
LSTM_LEARNING_RATE = 5e-4
# Residual anchor: "auto" picks linear vs naive by validation MAE; or force either mode.
LSTM_RESIDUAL_ANCHOR = "auto"

LSTM_TICKER_OVERRIDES: dict[str, dict] = {
    "TYRE": {"residual_anchor": "naive", "epochs": 120},
    "DIPD": {"residual_anchor": "linear", "epochs": 200, "tuning_epochs": 60},
}

LSTM_STREAM_FEATURES: dict[str, list[str]] = {
    "lstm_univariate": ["close_scaled"],
    "lstm_fundamental": [
        "close_scaled",
        "eps_scaled",
        "nav_per_share",
        "log_return",
        "rolling_ma_20",
        "rolling_std_20",
    ],
    "lstm_macro": ["close_scaled", "bond_rate_scaled"],
    "lstm_forex": ["close_scaled", "forex_close_scaled", "forex_log_return"],
}

# All trained LSTM streams available to the ensemble/meta-learner. lstm_macro
# (bond-rate-only features) is excluded here: across repeated ablation runs on
# TYRE and DIPD it is consistently and by far the weakest stream on the test
# split, yet its validation MAE is noisy enough to occasionally look
# deceptively strong, which would otherwise drag the ensemble/meta down. It
# remains in LSTM_TRAIN_ORDER and is still trained/reported standalone.
LSTM_META_STREAMS = ["lstm_univariate", "lstm_fundamental", "lstm_forex"]

# Day-t context concatenated to LSTM output (same signals linear regression uses).
LSTM_TABULAR_CONTEXT = [
    "close_scaled",
    "eps_scaled",
    "nav_per_share",
    "bond_rate_scaled",
    "forex_close_scaled",
    "rolling_ma_20",
    "rolling_std_20",
]
