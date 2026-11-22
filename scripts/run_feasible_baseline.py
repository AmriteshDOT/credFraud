import numpy as np
from src.ingestion.pit_batch import build_leak_free_dataset
from src.core.rewards import compute_fraud_reward
from src.eval.harness import EvaluationOracle, OOTFraudDataset
from src.models.baseline_fraud import CalibratedFraudBaseline
from src.models.baseline_credit import CMDPCreditBaseline
from src.models.coupled_baseline import CoupledEnterpriseBaseline

FEATURE_COLS = [
    "amount_inr",
    "is_device_missing",
    "tx_count_24h",
    "avg_amt_30d",
    "is_cold_start",
]


def fit_credit_baseline() -> CMDPCreditBaseline:
    """Fits pi_{C,0} on a 3-state (Current, DPD30, Default) x 3-action (None, SMS, Call) CMDP."""
    # States: 0=Current, 1=DPD_30, 2=Default (absorbing)
    # Actions: 0=No_Contact, 1=SMS_Nudge, 2=Agent_Call
    P_hat = np.array(
        [
            # From State 0 (Current)
            [[0.97, 0.03, 0.00], [0.98, 0.02, 0.00], [0.98, 0.02, 0.00]],
            # From State 1 (DPD_30)
            [[0.15, 0.70, 0.15], [0.30, 0.58, 0.12], [0.45, 0.45, 0.10]],
            # From State 2 (Default - absorbing)
            [[0.00, 0.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 1.00]],
        ],
        dtype=np.float64,
    )

    R_c = np.array(
        [
            [0.0, -2.0, -50.0],  # Current: contacting wastes money
            [500.0, 1200.0, 2200.0],  # DPD_30: expected daily recovery minus ops cost
            [0.0, -2.0, -50.0],  # Default: absorbing
        ],
        dtype=np.float64,
    )

    C_rbi = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    mu_0 = np.array([0.85, 0.15, 0.00], dtype=np.float64)

    pi_c0 = CMDPCreditBaseline(num_states=3, num_actions=3, c_max_rbi=15.0)
    return pi_c0.fit(P_hat=P_hat, R_c=R_c, C_rbi=C_rbi, mu_0=mu_0)


def main() -> None:
    # 1. Load leak-free, maturity-censored fraud dataset
    train_cutoff_ms = 1_777_593_600_000
    df = build_leak_free_dataset(
        tx_parquet_glob="data/raw/transactions.parquet",
        feat_parquet_glob="data/raw/features.parquet",
        chargeback_parquet_glob="data/raw/chargebacks.parquet",
        train_cutoff_ts_ms=train_cutoff_ms,
    ).sort("event_ts_ms")

    n = len(df)
    i_tr, i_ca = int(0.60 * n), int(0.80 * n)
    train_df, cal_df, oot_df = df[:i_tr], df[i_tr:i_ca], df[i_ca:]

    x_train, y_train = (
        train_df.select(FEATURE_COLS).to_numpy(),
        train_df["y_true"].to_numpy(),
    )
    x_cal, y_cal = cal_df.select(FEATURE_COLS).to_numpy(), cal_df["y_true"].to_numpy()
    x_oot, y_oot = oot_df.select(FEATURE_COLS).to_numpy(), oot_df["y_true"].to_numpy()

    # 2. Fit Phase I Fraud Policy pi_{F,0}
    pi_f0 = CalibratedFraudBaseline(
        amount_col_idx=0, alpha_conformal=0.05, max_fpr=0.005
    )
    pi_f0.fit(x_train, y_train, x_cal, y_cal)

    # 3. Fit Phase I Credit Policy pi_{C,0}
    pi_c0 = fit_credit_baseline()

    # 4. Assemble Unified Feasible Policy pi_0 = (pi_{F,0}, pi_{C,0})
    pi_0 = CoupledEnterpriseBaseline(fraud_policy=pi_f0, credit_policy=pi_c0)

    # 5. Verify Fraud Feasibility & Lock Benchmark J_0 via EvaluationOracle
    logged_actions = oot_df["logged_action"].to_numpy()
    propensities = oot_df["propensity"].to_numpy()
    amounts_oot = oot_df["amount_inr"].to_numpy()
    logged_rewards = compute_fraud_reward(logged_actions, y_oot, amounts_oot)

    p_oot = pi_f0.calibrator.predict(
        pi_f0.model.predict_proba(pi_f0._sanitize(x_oot))[:, 1]
    )
    q_hat = pi_f0._expected_utility(p_oot, amounts_oot, lam_fpr=0.0)

    oot_dataset = OOTFraudDataset(
        states=x_oot,
        logged_actions=logged_actions,
        logged_rewards=logged_rewards,
        y_true=y_oot,
        propensities=propensities,
        q_hat=q_hat,
    )
    stress_states = np.array(
        [
            [np.nan, 1.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1e8, 1.0, 500.0, 1e6, 0.0],
        ],
        dtype=np.float64,
    )

    oracle = EvaluationOracle(oot_data=oot_dataset, stress_states=stress_states)
    fraud_report = oracle.evaluate(pi_0.fraud_policy)

    assert (
        fraud_report.is_feasible
    ), "Phase I Fraud Baseline violated a hard constraint wall!"
    print(
        f"[Phase I Locked] Fraud J_F0 (95% LCB): INR {fraud_report.effective_j_inr:,.2f}"
    )
    print(
        f"[Phase I Locked] Fraud FPR: {fraud_report.fpr:.4%} | p99.9 Latency: {fraud_report.p999_latency_ms:.2f}ms"
    )
    print(
        f"[Phase I Locked] Credit J_C0 (LP NPV): INR {pi_0.credit_policy.optimal_npv_inr:,.2f}"
    )


if __name__ == "__main__":
    main()
