import numpy as np
from src.ingestion.pit_batch import build_leak_free_dataset
from src.core.rewards import compute_fraud_reward
from src.features.transforms import FEATURE_NAMES_V1, transform_batch
from src.eval.harness import EvaluationOracle, OOTFraudDataset
from src.models.baseline_fraud import CalibratedFraudBaseline
from src.models.cql_fraud import CQLFraudPolicy

MIN_RUPEE_LIFT_EPS = 250.0


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

    x_tr, y_tr = transform_batch(train_df), train_df["y_true"].to_numpy()
    x_ca, y_ca = transform_batch(cal_df), cal_df["y_true"].to_numpy()
    x_oot, y_oot = transform_batch(oot_df), oot_df["y_true"].to_numpy()

    logged_actions_oot = oot_df["logged_action"].to_numpy()
    propensities_oot = oot_df["propensity"].to_numpy()
    amounts_oot = oot_df["amount_inr"].to_numpy()
    logged_rewards_oot = compute_fraud_reward(logged_actions_oot, y_oot, amounts_oot)

    # 1. Fit Block 1 benchmark (GBDT + phi_1)
    pi_f0 = CalibratedFraudBaseline(
        amount_col_idx=0, alpha_conformal=0.05, max_fpr=0.005
    )
    pi_f0.fit(x_tr, y_tr, x_ca, y_ca)

    p_oot = pi_f0.calibrator.predict(
        pi_f0.model.predict_proba(pi_f0._sanitize(x_oot))[:, 1]
    )
    stress = np.zeros((3, len(FEATURE_NAMES_V1)), dtype=np.float64)
    stress[0, :] = np.nan
    stress[2, 0] = 1e8

    oracle = EvaluationOracle(
        oot_data=OOTFraudDataset(
            states=x_oot,
            logged_actions=logged_actions_oot,
            logged_rewards=logged_rewards_oot,
            y_true=y_oot,
            propensities=propensities_oot,
            q_hat=pi_f0._expected_utility(p_oot, amounts_oot, lam_fpr=0.0),
        ),
        stress_states=stress,
    )
    rep_f0 = oracle.evaluate(pi_f0)

    # 2. Fit Block 2 candidate (CQL + phi_1)
    r_train = compute_fraud_reward(
        action=train_df["logged_action"].to_numpy(),
        y_true=y_tr,
        amount_inr=train_df["amount_inr"].to_numpy(),
    )
    pi_f1 = CQLFraudPolicy(
        state_dim=x_tr.shape[1], gamma=0.90, alpha_cql=1.0, max_fpr=0.005
    )
    pi_f1.fit(
        payer_ids_train=train_df["payer_id"].to_numpy(),
        x_train=x_tr,
        actions_train=train_df["logged_action"].to_numpy(),
        rewards_train=r_train,
        x_cal=x_ca,
        y_cal=y_ca,
    )
    rep_f1 = oracle.evaluate(pi_f1)

    print(
        f"[GBDT + phi_1] Feasible: {rep_f0.is_feasible} | 95% LCB: INR {rep_f0.effective_j_inr:,.2f} | FPR: {rep_f0.fpr:.4%} | p99.9: {rep_f0.p999_latency_ms:.2f}ms"
    )
    print(
        f"[CQL  + phi_1] Feasible: {rep_f1.is_feasible} | 95% LCB: INR {rep_f1.effective_j_inr:,.2f} | FPR: {rep_f1.fpr:.4%} | p99.9: {rep_f1.p999_latency_ms:.2f}ms"
    )

    promoted = rep_f1.is_feasible and (
        rep_f1.effective_j_inr > rep_f0.effective_j_inr + MIN_RUPEE_LIFT_EPS
    )
    print(
        f"\n[GATE RESULT] Promote CQL (pi_F1): {promoted} (Delta LCB: INR {rep_f1.effective_j_inr - rep_f0.effective_j_inr:+,.2f})"
    )


if __name__ == "__main__":
    main()
