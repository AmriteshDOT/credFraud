import duckdb
import polars as pl


def build_leak_free_dataset(
    tx_parquet_glob: str,
    feat_parquet_glob: str,
    chargeback_parquet_glob: str,
    train_cutoff_ts_ms: int,
    maturity_window_ms: int = 45 * 86_400_000,  # 45-day chargeback maturity
) -> pl.DataFrame:
    """
    Handles massive batch volume (Parquet + DuckDB out-of-core),
    cold-start defaults (COALESCE), and right-censored labels (Y_cb).
    """
    mature_limit_ms = train_cutoff_ts_ms - maturity_window_ms

    sql = f"""
    SELECT
        tx.tx_id,
        tx.event_ts_ms,
        tx.amount_inr,
        CASE WHEN tx.device_id IS NULL THEN 1 ELSE 0 END AS is_device_missing,
        COALESCE(feat.tx_count_24h, 0)                   AS tx_count_24h,
        COALESCE(feat.avg_amt_30d, -1.0)                 AS avg_amt_30d,
        CASE WHEN feat.snapshot_ts_ms IS NULL THEN 1 ELSE 0 END AS is_cold_start,
        COALESCE(cb.is_fraud, 0)                         AS y_true
    FROM read_parquet('{tx_parquet_glob}') AS tx
    ASOF LEFT JOIN read_parquet('{feat_parquet_glob}') AS feat
        ON tx.payer_id = feat.payer_id
       AND tx.event_ts_ms > feat.snapshot_ts_ms          -- Strict causal < t
    LEFT JOIN read_parquet('{chargeback_parquet_glob}') AS cb
        ON tx.tx_id = cb.tx_id
       AND cb.classification_ts_ms <= {train_cutoff_ts_ms}
    WHERE tx.event_ts_ms <= {mature_limit_ms}            -- Drop immature right-censored rows
    """
    return duckdb.sql(sql).pl()
