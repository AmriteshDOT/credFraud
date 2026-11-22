import numpy as np
from src.ingestion.pit_batch import build_leak_free_dataset
from src.core.rewards import compute_fraud_reward
from src.features.transforms import FEATURE_NAMES_V1, transform_batch
from src.eval.harness import EvaluationOracle, OOTFraudDataset
from src.models.baseline_fraud import CalibratedFraudBaseline

BASE_COLS_V0 = [
    "amount_inr",
    "is_device_missing",
    "tx_count_24h",
    "avg_amt_30d",
    "is_cold_start",
]
MIN_RUPEE_LIFT_EPS = 250.0


def build_stress_matrix(num_cols: int) -> np.ndarray:
    stress = np.zeros((3, num_cols), dtype=np.float64)
    stress[0, :] = np.nan
    stress[2, 0] = 1e8
    return stress


def evaluate_representation(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ca: np.ndarray,
    y_ca: np.ndarray,
    x_oot: np.ndarray,
    y_oot: np.ndarray,
    logged_actions: np.ndarray,
    logged_rewards: np.ndarray,
    propensities: np.ndarray,
    amounts_oot: np.ndarray,
) -> tuple[CalibratedFraudBaseline, object]:
    policy = CalibratedFraudBaseline(
        amount_col_idx=0, alpha_conformal=0.05, max_fpr=0.005
    )
    policy.fit(x_tr, y_tr, x_ca, y_ca)

    p_oot = policy.calibrator.predict(
        policy.model.predict_proba(policy._sanitize(x_oot))[:, 1]
    )
    q_hat = policy._expected_utility(p_oot, amounts_oot, lam_fpr=0.0)

    oracle = EvaluationOracle(
        oot_data=OOTFraudDataset(
            states=x_oot,
            logged_actions=logged_actions,
            logged_rewards=logged_rewards,
            y_true=y_oot,
            propensities=propensities,
            q_hat=q_hat,
        ),
        stress_states=build_stress_matrix(x_tr.shape[1]),
    )
    return policy, oracle.evaluate(policy)


def main() -> None:
    df = build_leak_free_dataset(
        tx_parquet_glob="data/raw/transactions.parquet",
        feat_parquet_glob="data/raw/features.parquet",
        chargeback_parquet_glob="data/raw/chargebacks.parquet",
        train_cutoff_ts_ms=1_777_593_600_000,
    ).sort("event_ts_ms")

    n = len(df)
    i_tr, i_ca = int(0.60 * n), int(0.80 * n)
    train_df, cal_df, oot_df = df[:i_tr], df[i_tr:i_ca], df[i_ca:]

    y_tr, y_ca, y_oot = (
        train_df["y_true"].to_numpy(),
        cal_df["y_true"].to_numpy(),
        oot_df["y_true"].to_numpy(),
    )
    logged_actions = oot_df["logged_action"].to_numpy()
    propensities = oot_df["propensity"].to_numpy()
    amounts_oot = oot_df["amount_inr"].to_numpy()
    logged_rewards = compute_fraud_reward(logged_actions, y_oot, amounts_oot)

    # Phase I Anchor: pi_{F,0} o phi_0
    pi_v0, rep_v0 = evaluate_representation(
        train_df.select(BASE_COLS_V0).to_numpy(),
        y_tr,
        cal_df.select(BASE_COLS_V0).to_numpy(),
        y_ca,
        oot_df.select(BASE_COLS_V0).to_numpy(),
        y_oot,
        logged_actions,
        logged_rewards,
        propensities,
        amounts_oot,
    )

    # Phase II Block 1: pi_{F,0} o phi_1
    pi_v1, rep_v1 = evaluate_representation(
        transform_batch(train_df),
        y_tr,
        transform_batch(cal_df),
        y_ca,
        transform_batch(oot_df),
        y_oot,
        logged_actions,
        logged_rewards,
        propensities,
        amounts_oot,
    )

    print(
        f"[phi_0] Feasible: {rep_v0.is_feasible} | 95% LCB: INR {rep_v0.effective_j_inr:,.2f} | FPR: {rep_v0.fpr:.4%} | lam_FPR*: {pi_v0.lambda_fpr:.2f}"
    )
    print(
        f"[phi_1] Feasible: {rep_v1.is_feasible} | 95% LCB: INR {rep_v1.effective_j_inr:,.2f} | FPR: {rep_v1.fpr:.4%} | lam_FPR*: {pi_v1.lambda_fpr:.2f}"
    )

    promoted = rep_v1.is_feasible and (
        rep_v1.effective_j_inr > rep_v0.effective_j_inr + MIN_RUPEE_LIFT_EPS
    )
    print(
        f"\n[GATE RESULT] Promote phi_1: {promoted} (Delta LCB: INR {rep_v1.effective_j_inr - rep_v0.effective_j_inr:+,.2f})"
    )


if __name__ == "__main__":
    main()
