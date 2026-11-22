from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

IST = timezone(timedelta(hours=5, minutes=30))
ACTION_MAP_FRAUD = {0: "Approve", 1: "2FA", 2: "Soft_Hold", 3: "Hard_Block"}
ACTION_MAP_CREDIT = {0: "No_Contact", 1: "SMS_Nudge", 2: "Agent_Call"}


class FraudRequest(BaseModel):
    tx_id: str
    payer_id: str
    amount_inr: float = Field(ge=0.0)
    is_device_missing: int | None = None
    tx_count_24h: float | None = None
    tx_count_1h: float | None = None
    avg_amt_30d: float | None = None
    is_cold_start: int | None = None
    event_ts_ms: int
    device_account_degree: float | None = None


class FraudResponse(BaseModel):
    tx_id: str
    action_id: int
    action_name: str
    hitl_required: bool
    propensity: float
    latency_ms: float


class CreditRequest(BaseModel):
    loan_id: str
    delinquency_state: int = Field(ge=0, le=2)  # 0=Current, 1=DPD_30, 2=Default
    contacts_this_month: int = Field(ge=0)
    max_monthly_contacts: int = 15


class CreditResponse(BaseModel):
    loan_id: str
    action_id: int
    action_name: str
    rbi_override_applied: bool


def project_fraud_rbi(raw_action: int, req: FraudRequest) -> tuple[int, bool]:
    """
    Deterministic projection P_RBI^F:
    Enforces Natural Justice HITL on Hard_Block (3) for cold-start or high-value accounts.
    """
    if raw_action == 3 and (req.is_cold_start == 1 or req.amount_inr >= 100_000.0):
        return 2, True  # Downgrade Hard_Block -> Soft_Hold with mandatory HITL review
    return raw_action, (raw_action == 2)


def project_credit_rbi(raw_action: int, req: CreditRequest) -> tuple[int, bool]:
    """
    Deterministic projection P_RBI^C:
    Enforces RBI Fair Practices DND window (08:00-19:00 IST) and monthly contact cap.
    """
    now_ist_hour = datetime.now(IST).hour
    in_dnd_window = not (8 <= now_ist_hour < 19)
    cap_reached = req.contacts_this_month >= req.max_monthly_contacts

    if raw_action != 0 and (in_dnd_window or cap_reached):
        return 0, True  # Force No_Contact
    return raw_action, False
