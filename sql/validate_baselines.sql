-- Compare baseline model results for a ticker

USE equity_compass;

SELECT model_type, mae, rmse, mape, r2, directional_accuracy,
       test_from, test_to, trained_at
FROM model_runs
WHERE ticker = 'TYRE'
ORDER BY mae;

SELECT model_type, split_flag, COUNT(*) AS predictions
FROM predictions p
JOIN model_runs m ON m.run_id = p.run_id
WHERE p.ticker = 'TYRE'
GROUP BY model_type, split_flag
ORDER BY model_type, split_flag;
