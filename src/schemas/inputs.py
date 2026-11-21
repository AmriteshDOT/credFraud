from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PaymentInput(BaseModel):
    """Total ingress parser for X_tx and cold-start edge cases (X_cold)."""

    tx_id: str = Field(min_length=1)
    event_ts_ms: int = Field(gt=0)
    payer_id: str = Field(min_length=1)
    payee_id: str = Field(min_length=1)
    amount_inr: float = Field(gt=0.0, le=1_000_000_000.0)
    rail: Literal["UPI", "IMPS", "NEFT", "Card"]
    device_id: Optional[str] = None  # Explicit bot/null support
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    remark: str = Field(default="", max_length=256)

    @field_validator("device_id", mode="before")
    @classmethod
    def normalize_empty_device(cls, v):
        # Coerce messy "", "null", "N/A" strings into canonical Python None
        if v is None or str(v).strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return str(v).strip()


class CreditPanelInput(BaseModel):
    """Total ingress parser for daily borrower state X_credit."""

    loan_id: str
    day_tau: int = Field(ge=0, le=180)
    dpd: int = Field(ge=0)
    ead_inr: float = Field(ge=0.0)
    emi_inr: float = Field(gt=0.0)
    contacts_7d: int = Field(ge=0)
    is_event_observed: Literal[0, 1]  # Survival censoring flag delta_{i, tau}
