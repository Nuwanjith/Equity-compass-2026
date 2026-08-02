# Equity Compass 2026

Machine-learning pipeline for **30-day-ahead stock price forecasting** on Colombo Stock Exchange (CSE) tickers. Trains baselines and LSTM ensembles, logs metrics to MySQL, generates evaluation reports, and feeds the **Equity-Compass-Web** PHP frontend.

**Forecast horizon:** 30 trading days · **Train/val/test split:** 70% / 15% / 15%

---

## Pipeline at a glance

```
DATA/ (CSV)  →  MySQL raw tables  →  preprocess  →  features_daily
                                                          ↓
                    train_baselines + train_models  →  model_runs + predictions
                                                          ↓
                    evaluate  →  reports/          forecast_future  →  future predictions
                    compute_valuations  →  valuations table
```

| Phase | Script | Output |
|-------|--------|--------|
| Import | `scripts/import/import_csv.py` | `raw_stock_prices`, `raw_fundamentals`, etc. |
| Preprocess | `scripts/preprocess.py` | `features_daily`, `scaler_params` |
| Baselines | `scripts/train_baselines.py` | `model_runs`, `predictions` |
| LSTM + meta | `scripts/train_models.py` | Saved `.keras` / `.json` in `models/`, DB rows |
| Evaluate | `scripts/evaluate.py` | `reports/{TICKER}/report.md`, CSVs, charts |
| Future forecast | `scripts/forecast_future.py` | `predictions` with `split_flag='future'` |
| Valuation | `scripts/compute_valuations.py` | `valuations` (NAV, EPS×10.35, Graham) |

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure MySQL (copy and edit)
cp config/.env.example config/.env

# 3. Create DB + model tables (see sql/)
mysql -u root -p equity_compass < sql/create_model_tables.sql

# 4. Run pipeline (per ticker; DIPD & TYRE are fully trained with fundamentals)
python scripts/preprocess.py --ticker DIPD
python scripts/train_baselines.py --ticker DIPD
python scripts/train_models.py --ticker DIPD
python scripts/evaluate.py --ticker DIPD
python scripts/forecast_future.py --ticker DIPD
python scripts/compute_valuations.py --ticker DIPD
```

---

## Critical components

### Configuration — `src/equity_compass/config.py`

Single source of truth: DB credentials, `TICKERS` (11 symbols), split ratios, LSTM hyperparameters, feature sets per stream, corporate-action rules (DIPD 10-for-1 split), and ensemble stream selection (`LSTM_META_STREAMS` excludes weak `lstm_macro`).

### Preprocessing — `src/equity_compass/preprocessing/`

| Module | Role |
|--------|------|
| `pipeline.py` | Joins prices, forex, bonds, fundamentals → engineered features, MinMax scaling, train/val/test flags |
| `corporate_actions.py` | Back-adjusts pre-split prices (DIPD) |

Reads: `raw_stock_prices`, `raw_forex_rates`, `raw_bond_rates`, `raw_fundamentals`  
Writes: `features_daily`, `scaler_params`

### Training — `src/equity_compass/training/`

| Module | Role |
|--------|------|
| `data.py` | Load features, date ranges per split |
| `baselines.py` | Naive, linear regression, ARIMA, ARIMAX, XGBoost tabular |
| `lstm_models.py` | Four LSTM streams, weighted ensemble, XGBoost meta-learner |
| `sequences.py` | Leakage-safe sliding windows (lookback 60) |
| `metrics.py` | MAE, RMSE, MAPE, R², directional accuracy |
| `registry.py` | **`evaluate_and_save()`** — computes test metrics, writes `model_runs` + `predictions` |
| `forecast.py` | Reuses saved artifacts for out-of-sample future rows |

**Model stack (in training order):**

1. **Baselines** — naive, linear_regression, arima, arimax, xgboost_tabular  
2. **LSTM streams** — univariate, fundamental, macro, forex (residual target vs naive/linear anchor)  
3. **Ensemble** — `lstm_avg_ensemble` (validation-weighted blend)  
4. **Meta** — `xgboost_meta` (blends LSTM streams; trained on val only)

Artifacts saved under `models/{TICKER}/{model_type}/`.

### Evaluation — `src/equity_compass/evaluation/`

| Module | Role |
|--------|------|
| `loaders.py` | Read `model_runs` and `predictions` from MySQL |
| `report.py` | Leaderboard, ablation tables, monthly errors, markdown report |
| `charts.py` | MAE comparison, predicted vs actual, monthly MAE plots |

**Metrics flow:** Headline numbers (one MAE/RMSE/MAPE/R² per model) are computed at **training time** in `registry.evaluate_and_save()` → stored in `model_runs`. `evaluate.py` reads and publishes them; it does not recompute the leaderboard.

### Valuation — `src/equity_compass/valuation.py`

NAV per share, EPS × 10.35 (CSE P/E proxy), and Graham Number. Requires quarterly fundamentals (currently **DIPD** and **TYRE** only).

### Database — `src/equity_compass/database.py`

SQLAlchemy engine from `config/.env`. Schema DDL in `sql/`.

---

## Key database tables

| Table | Purpose |
|-------|---------|
| `raw_stock_prices` | Daily OHLCV |
| `raw_fundamentals` | Quarterly EPS, NAV |
| `raw_forex_rates` / `raw_bond_rates` | Macro inputs |
| `features_daily` | Preprocessed feature matrix + `split_flag` |
| `scaler_params` | MinMax scaler bounds per ticker |
| `model_runs` | One row per (ticker, model_type) with aggregate metrics |
| `predictions` | Daily forecast rows (`train` / `val` / `test` / `future`) |
| `valuations` | Latest fundamental valuation snapshot |

---

## Metrics (test split)

| Metric | Meaning |
|--------|---------|
| **MAE** | Average absolute price error (Rs) |
| **RMSE** | Root mean squared error — penalises large misses |
| **MAPE** | Mean absolute percentage error |
| **R²** | Variance explained vs mean of actual test prices |
| **Directional accuracy** | % of correct up/down moves day-to-day |

---

## Project layout

```
Equity-compass-2026/
├── config/.env.example      # DB credentials template
├── DATA/                    # Source CSVs (prices, EPS/NAV, forex, bonds)
├── models/                  # Trained LSTM + XGBoost artifacts
├── reports/                 # Generated evaluation output
├── scripts/                 # CLI entry points (run these)
├── sql/                     # DDL + validation queries
└── src/equity_compass/      # Core library
    ├── config.py
    ├── database.py
    ├── valuation.py
    ├── preprocessing/
    ├── training/
    └── evaluation/
```

---

## Tickers & data coverage

Configured in `config.TICKERS`: CARG, COMB, CTC, DIPD, HAYC, HAYL, KCAB, KVAL, MELS, SAMP, TYRE.

**Fully pipeline-ready (fundamentals + trained models):** DIPD, TYRE.  
Other tickers can be preprocessed and baseline-trained; LSTM fundamental stream needs `raw_fundamentals` rows.

---

## Web frontend

The PHP dashboard lives in a separate **Equity-Compass-Web** repo. It reads from the same `equity_compass` MySQL database:

- `stock_prices.php` → `raw_stock_prices`
- `stock_predictions.php` → `predictions` + `model_runs` (`xgboost_meta`)
- `stock_valuations.php` → `valuations`

Run `forecast_future.py` after training so prediction charts extend past the last historical month.

---

## Design notes (thesis-relevant)

- **Residual targets:** LSTMs predict deviation from a naive or linear anchor, not raw price.
- **No leakage:** Sequence windows respect split boundaries; meta-learner fits on validation only.
- **Ensemble pruning:** `lstm_macro` is trained and reported but excluded from ensemble/meta blend.
- **Corporate actions:** DIPD prices pre-2021 are back-adjusted for the 10-for-1 subdivision.

---

## Dependencies

Python 3.10+, MySQL 8. See `requirements.txt` (pandas, scikit-learn, statsmodels, xgboost, tensorflow, SQLAlchemy, matplotlib).
