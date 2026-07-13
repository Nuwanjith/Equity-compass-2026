-- Validation queries after running scripts/preprocess.py

USE equity_compass;

-- Row counts per ticker and split
SELECT ticker, split_flag, COUNT(*) AS rows,
       MIN(feature_date) AS from_date,
       MAX(feature_date) AS to_date
FROM features_daily
GROUP BY ticker, split_flag
ORDER BY ticker, split_flag;

-- Scalers per ticker
SELECT ticker, feature_name, scaler_type, param_min, param_max, fitted_on
FROM scaler_params
ORDER BY ticker, feature_name;

-- Sample TYRE rows
SELECT feature_date, close_price, close_scaled, eps, eps_scaled,
       bond_rate, forex_close, target_close_30d, split_flag
FROM features_daily
WHERE ticker = 'TYRE'
ORDER BY feature_date DESC
LIMIT 10;
