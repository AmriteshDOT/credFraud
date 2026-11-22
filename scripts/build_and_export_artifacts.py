from pathlib import Path
import numpy as np
from src.ingestion.pit_batch import build_leak_free_dataset
from src.core.rewards import compute_fraud_reward
from src.features.transforms import transform_batch
from src.models.cql_fraud import CQLFraudPolicy
from src.serving.export_onnx import ONNXFraudRuntime, export_cql_to_onnx

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load leak-free dataset & split chronologically
    df = build_leak_free_dataset(
        tx_parquet_glob="data/raw/transactions.parquet",
        feat_parquet_glob="data/raw/features.parquet",
        chargeback_parquet_glob="data/raw/chargebacks.parquet",
        train_cutoff_ts_ms=1_777_593_600_000,
    ).sort("event_ts_ms")

    n = len(df)
    i_tr, i_ca = int(0.60 * n), int(0.80 * n)
    train_df, cal_df = df[:i_tr], df[i_tr:i_ca]

    x_tr = transform_batch(train_df)
    x_ca = transform_batch(cal_df)
    y_tr = train_df["y_true"].to_numpy()
    y_ca = cal_df["y_true"].to_numpy()
    r_tr = compute_fraud_reward(
        action=train_df["logged_action"].to_numpy(),
        y_true=y_tr,
        amount_inr=train_df["amount_inr"].to_numpy(),
    )

    # 2. Train CQLFraudPolicy (pi_{F,1})
    pi_f1 = CQLFraudPolicy(
        state_dim=x_tr.shape[1], gamma=0.90, alpha_cql=1.0, max_fpr=0.005
    )
    pi_f1.fit(
        payer_ids_train=train_df["payer_id"].to_numpy(),
        x_train=x_tr,
        actions_train=train_df["logged_action"].to_numpy(),
        rewards_train=r_tr,
        x_cal=x_ca,
        y_cal=y_ca,
    )

    # 3. Export ONNX graph + frozen normalization & dual parameters
    onnx_path = str(ARTIFACTS_DIR / "cql_fraud.onnx")
    export_cql_to_onnx(pi_f1, output_path=onnx_path)
    np.save(ARTIFACTS_DIR / "cql_mean.npy", pi_f1.mean_)
    np.save(ARTIFACTS_DIR / "cql_std.npy", pi_f1.std_)
    np.save(
        ARTIFACTS_DIR / "cql_lambda_fpr.npy",
        np.array(pi_f1.lambda_fpr, dtype=np.float64),
    )

    # 4. Pre-flight latency verification against the 35ms SLA
    runtime = ONNXFraudRuntime(
        onnx_path=onnx_path,
        mean=pi_f1.mean_,
        std=pi_f1.std_,
        lambda_fpr=pi_f1.lambda_fpr,
    )
    p999_ms = runtime.benchmark_p999_ms(x_ca[:1], n_runs=2000)
    assert p999_ms <= 35.0, f"SLA breach: p99.9 latency {p999_ms:.2f}ms > 35ms"
    print(
        f"[EXPORT COMPLETE] Saved to {ARTIFACTS_DIR}/ | ONNX p99.9 Latency: {p999_ms:.3f}ms"
    )


if __name__ == "__main__":
    main()
