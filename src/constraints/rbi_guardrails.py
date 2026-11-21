from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from pydantic import BaseModel, Field, model_validator

IST = ZoneInfo("Asia/Kolkata")


class CollectionActionGate(BaseModel):
    """Enforces RBI Curfew [08:00, 19:00) IST, Weekly Contact Cap, and AI Disclosure."""

    action: Literal["WAIT", "WHATSAPP_NUDGE", "GENAI_NEGOTIATE", "FIELD_VISIT"]
    timestamp_utc: datetime
    contacts_last_7d: int = Field(ge=0)
    weekly_cap: int = Field(default=3, ge=1)
    waiver_pct: float = Field(ge=0.0, le=1.0)
    max_cmdp_waiver_pct: float = Field(ge=0.0, le=1.0)
    ai_disclosed: bool = True
    harassment_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def verify_rbi_compliance(self) -> "CollectionActionGate":
        is_contact = self.action != "WAIT"
        ist_hour = self.timestamp_utc.astimezone(IST).hour

        if is_contact and not (8 <= ist_hour < 19):
            raise ValueError(f"RBI Curfew Violation: contact at {ist_hour}:00 IST")
        if is_contact and self.contacts_last_7d >= self.weekly_cap:
            raise ValueError(
                f"RBI Frequency Cap Violation: {self.contacts_last_7d} >= {self.weekly_cap}"
            )
        if is_contact and not self.ai_disclosed:
            raise ValueError("RBI Disclosure Violation: AI identity must be disclosed")
        if self.harassment_score > 0.0:
            raise ValueError(
                "RBI Fair Practices Violation: non-zero harassment detected"
            )
        if self.waiver_pct > self.max_cmdp_waiver_pct:
            raise ValueError(
                f"Solvency Violation: waiver {self.waiver_pct} > CMDP bound {self.max_cmdp_waiver_pct}"
            )
        return self


def determine_rbi_escalation(
    is_confirmed_fraud: bool, amount_inr: float
) -> dict[str, bool]:
    """Enforces RBI 2026 LEA/SFIO Reporting & Natural Justice HITL Gate."""
    if not is_confirmed_fraud:
        return {
            "report_lea_police": False,
            "report_sfio_fmr": False,
            "require_hitl": False,
        }

    return {
        "report_lea_police": amount_inr >= 100_000.0,  # >= 1 Lakh INR
        "report_sfio_fmr": amount_inr >= 10_000_000.0,  # >= 1 Crore INR
        "require_hitl": True,  # Natural Justice: human sign-off for civil freeze
    }
