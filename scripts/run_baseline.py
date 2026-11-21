import numpy as np
from src.ingestion.pit_batch import build_leak_free_dataset
from src.core.rewards import compute_fraud_reward
from src.models.baseline_fraud import CalibratedFraudBaseline
from src.eval.harness import EvaluationOracle, OOTFraudDataset

FEATURE_COLS = [
    "amount_inr",
    "is_device_missing",
    "tx_count_24h",
    "avg_amt_30d",
    "is_cold_start",
]


def main() -> None:
    # 1. Load point-in-time, maturity-censored dataset via DuckDB -> Polars
    train_cutoff_ms = 1_777_593_600_000  # 2026-05-01 00:00:00 UTC
    df = build_leak_free_dataset(
        tx_parquet_glob="data/raw/transactions.parquet",
        feat_parquet_glob="data/raw/features.parquet",
        chargeback_parquet_glob="data/raw/chargebacks.parquet",
        train_cutoff_ts_ms=train_cutoff_ms,
    ).sort("event_ts_ms")

    # 2. Strict Chronological Split: 60% D_train | 20% D_cal | 20% D_OOT
    n = len(df)
    i_train = int(0.60 * n)
    i_cal = int(0.80 * n)

    train_df = df[:i_train]
    cal_df = df[i_train:i_cal]
    oot_df = df[i_cal:]

    x_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["y_true"].to_numpy()

    x_cal = cal_df.select(FEATURE_COLS).to_numpy()
    y_cal = cal_df["y_true"].to_numpy()

    x_oot = oot_df.select(FEATURE_COLS).to_numpy()
    y_oot = oot_df["y_true"].to_numpy()

    # 3. Fit & Calibrate Baseline Policy pi_{F,0} strictly on D_train and D_cal
    pi_f0 = CalibratedFraudBaseline(
        amount_col_idx=0, alpha_conformal=0.05, max_fpr=0.005
    )
    pi_f0.fit(x_train, y_train, x_cal, y_cal)

    # 4. Build Frozen OOT Evaluation Dataset (D_OOT) and Stress Set (D_stress)
    # Note: logged_action & propensity come from transactions.parquet
    logged_actions = oot_df["logged_action"].to_numpy()
    propensities = oot_df["propensity"].to_numpy()
    amounts_oot = oot_df["amount_inr"].to_numpy()

    logged_rewards = compute_fraud_reward(
        action=logged_actions,
        y_true=y_oot,
        amount_inr=amounts_oot,
    )

    # Direct-method nuisance matrix Q_hat [N_oot, 4] using calibrated baseline prior
    p_oot = pi_f0.calibrator.predict(pi_f0.model.predict_proba(x_oot)[:, 1])
    q_hat = pi_f0._expected_utility(p_oot, amounts_oot, lam_fpr=0.0)

    oot_dataset = OOTFraudDataset(
        states=x_oot,
        logged_actions=logged_actions,
        logged_rewards=logged_rewards,
        y_true=y_oot,
        propensities=propensities,
        q_hat=q_hat,
    )

    # Stress inputs: cold-start NaNs, zero amounts, extreme rupee values
    stress_states = np.array(
        [
            [np.nan, 1.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1e8, 1.0, 500.0, 1e6, 0.0],
        ],
        dtype=np.float64,
    )

    # 5. Run Decoupled Evaluation Oracle O(pi_{F,0})
    oracle = EvaluationOracle(oot_data=oot_dataset, stress_states=stress_states)
    baseline_report = oracle.evaluate(pi_f0)

    assert baseline_report.is_feasible, "Baseline violated a hard constraint wall!"
    print(f"Locked Benchmark J_0 (95% LCB): INR {baseline_report.effective_j_inr:,.2f}")
    print(f"DR Mean: INR {baseline_report.dr_mean_inr:,.2f}")
    print(
        f"FPR: {baseline_report.fpr:.4%} | p99.9 Latency: {baseline_report.p999_latency_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
