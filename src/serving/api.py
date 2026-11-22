import json
import time
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from src.features.transforms import transform_single
from src.serving.export_onnx import ONNXFraudRuntime
from src.serving.guardrails import (
    ACTION_MAP_CREDIT,
    ACTION_MAP_FRAUD,
    CreditRequest,
    CreditResponse,
    FraudRequest,
    FraudResponse,
    project_credit_rbi,
    project_fraud_rbi,
)
from scripts.run_feasible_baseline import fit_credit_baseline

app = FastAPI(title="Kotak FraudShield & Credit Risk Engine", version="1.0.0")

AUDIT_LOG_PATH = Path("data/logs/live_propensity_audit.jsonl")
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load frozen Phase II artifacts at startup (zero per-request initialization overhead)
fraud_runtime = ONNXFraudRuntime(
    onnx_path="artifacts/cql_fraud.onnx",
    mean=np.load("artifacts/cql_mean.npy"),
    std=np.load("artifacts/cql_std.npy"),
    lambda_fpr=float(np.load("artifacts/cql_lambda_fpr.npy")),
)
credit_policy = fit_credit_baseline()


def _log_audit(record: dict) -> None:
    """Append-only closed-loop propensity log for future DR-OPE & CQL retraining."""
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/v1/fraud/score", response_model=FraudResponse)
def score_fraud(req: FraudRequest) -> FraudResponse:
    t0 = time.perf_counter_ns()

    # 1. Zero-skew online feature transform phi_1(x_t)
    x_vec = transform_single(req.model_dump())

    # 2. Sub-millisecond ONNX CQL inference pi_F*
    raw_action = fraud_runtime.predict_action(x_vec)

    # 3. Deterministic Natural Justice & RBI projection P_RBI^F
    final_action, hitl_required = project_fraud_rbi(raw_action, req)

    latency_ms = (time.perf_counter_ns() - t0) / 1e6

    # 4. Closed-loop propensity logging
    _log_audit(
        {
            "tx_id": req.tx_id,
            "payer_id": req.payer_id,
            "event_ts_ms": req.event_ts_ms,
            "action_id": final_action,
            "propensity": 1.0,
            "latency_ms": round(latency_ms, 3),
        }
    )

    return FraudResponse(
        tx_id=req.tx_id,
        action_id=final_action,
        action_name=ACTION_MAP_FRAUD[final_action],
        hitl_required=hitl_required,
        propensity=1.0,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/v1/credit/decide", response_model=CreditResponse)
def decide_credit(req: CreditRequest) -> CreditResponse:
    probs = credit_policy.predict_proba(np.array([req.delinquency_state]))[0]
    raw_action = int(np.random.choice(len(probs), p=probs))
    final_action, overridden = project_credit_rbi(raw_action, req)

    return CreditResponse(
        loan_id=req.loan_id,
        action_id=final_action,
        action_name=ACTION_MAP_CREDIT[final_action],
        rbi_override_applied=overridden,
    )
