import math
from typing import Mapping
import numpy as np
import polars as pl

FEATURE_NAMES_V1: tuple[str, ...] = (
    "amount_inr",
    "log_amount",
    "amt_to_avg_30d_ratio",
    "is_device_missing",
    "tx_count_24h",
    "tx_velocity_ratio",
    "avg_amt_30d",
    "is_cold_start",
    "hour_sin_ist",
    "hour_cos_ist",
    "device_account_degree",
)

IST_OFFSET_MS: int = 19_800_000
DAY_MS: int = 86_400_000


def transform_batch(df: pl.DataFrame) -> np.ndarray:
    """Offline batch operator phi_1(X; H_<t) -> [N, 11] float64 matrix (NaN-safe)."""
    amt = pl.col("amount_inr").fill_null(0.0).clip(lower_bound=0.0)
    avg_30d = pl.col("avg_amt_30d").fill_null(-1.0)
    tx_24h = pl.col("tx_count_24h").fill_null(0.0).clip(lower_bound=0.0)
    tx_1h = (
        pl.col("tx_count_1h").fill_null(0.0).clip(lower_bound=0.0)
        if "tx_count_1h" in df.columns
        else pl.lit(0.0)
    )
    dev_deg = (
        pl.col("device_account_degree").fill_null(0.0).clip(lower_bound=0.0)
        if "device_account_degree" in df.columns
        else pl.lit(0.0)
    )

    hour_ist = (
        (pl.col("event_ts_ms").fill_null(0) + IST_OFFSET_MS) % DAY_MS
    ) / 3_600_000.0
    angle = (2.0 * math.pi * hour_ist) / 24.0

    return (
        df.select(
            [
                amt.alias("amount_inr"),
                amt.log1p().alias("log_amount"),
                (amt / avg_30d.clip(lower_bound=1.0)).alias("amt_to_avg_30d_ratio"),
                pl.col("is_device_missing").fill_null(1.0).cast(pl.Float64),
                tx_24h.alias("tx_count_24h"),
                (tx_1h / (tx_24h / 24.0).clip(lower_bound=0.1)).alias(
                    "tx_velocity_ratio"
                ),
                avg_30d.alias("avg_amt_30d"),
                pl.col("is_cold_start").fill_null(1.0).cast(pl.Float64),
                angle.sin().alias("hour_sin_ist"),
                angle.cos().alias("hour_cos_ist"),
                dev_deg.alias("device_account_degree"),
            ]
        )
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def transform_single(event: Mapping[str, float | int | None]) -> np.ndarray:
    """Online single-event operator phi_1(x_t; H_<t) -> [1, 11] float64 matrix."""
    amt = max(float(event.get("amount_inr") or 0.0), 0.0)
    avg_30d = (
        float(event["avg_amt_30d"]) if event.get("avg_amt_30d") is not None else -1.0
    )
    tx_24h = max(float(event.get("tx_count_24h") or 0.0), 0.0)
    tx_1h = max(float(event.get("tx_count_1h") or 0.0), 0.0)
    dev_deg = max(float(event.get("device_account_degree") or 0.0), 0.0)
    is_dev_missing = float(
        event.get("is_device_missing")
        if event.get("is_device_missing") is not None
        else 1.0
    )
    is_cold = float(
        event.get("is_cold_start") if event.get("is_cold_start") is not None else 1.0
    )

    ts_ms = int(event.get("event_ts_ms") or 0)
    hour_ist = ((ts_ms + IST_OFFSET_MS) % DAY_MS) / 3_600_000.0
    angle = (2.0 * math.pi * hour_ist) / 24.0

    return np.array(
        [
            [
                amt,
                math.log1p(amt),
                amt / max(avg_30d, 1.0),
                is_dev_missing,
                tx_24h,
                tx_1h / max(tx_24h / 24.0, 0.1),
                avg_30d,
                is_cold,
                math.sin(angle),
                math.cos(angle),
                dev_deg,
            ]
        ],
        dtype=np.float64,
    )
