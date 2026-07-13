-- Phase 4 tables: model training logs and predictions

USE equity_compass;

CREATE TABLE IF NOT EXISTS model_runs (
    run_id               INT AUTO_INCREMENT PRIMARY KEY,
    ticker               VARCHAR(10)  NOT NULL,
    model_type           ENUM(
        'naive',
        'arima',
        'arimax',
        'linear_regression',
        'xgboost_tabular',
        'lstm_univariate',
        'lstm_fundamental',
        'lstm_macro',
        'lstm_forex',
        'lstm_avg_ensemble',
        'xgboost_meta'
    ) NOT NULL,
    trained_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    train_from           DATE NULL,
    train_to             DATE NULL,
    val_from             DATE NULL,
    val_to               DATE NULL,
    test_from            DATE NULL,
    test_to              DATE NULL,
    mae                  FLOAT NULL,
    rmse                 FLOAT NULL,
    mape                 FLOAT NULL,
    r2                   FLOAT NULL,
    directional_accuracy FLOAT NULL,
    model_path           VARCHAR(500) NULL,
    notes                TEXT NULL,
    INDEX idx_ticker_model (ticker, model_type)
);

CREATE TABLE IF NOT EXISTS predictions (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id           INT          NOT NULL,
    ticker           VARCHAR(10)  NOT NULL,
    predicted_at     DATE         NOT NULL,
    target_date      DATE         NOT NULL,
    predicted_price  DECIMAL(12,4) NOT NULL,
    actual_price     DECIMAL(12,4) NULL,
    split_flag       ENUM('train','val','test') NOT NULL,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    INDEX idx_ticker_target (ticker, target_date),
    INDEX idx_run_id (run_id)
);
