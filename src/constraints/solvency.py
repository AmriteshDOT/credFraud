import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolvencyReport:
    fpr: float
    expected_loss_inr: float
    fpr_compliant: bool
    basel_compliant: bool


def check_solvency_constraints(
    fraud_actions: np.ndarray,  # 0 = Approve, != 0 = Intervene
    fraud_labels: np.ndarray,  # 0 = Legit, 1 = Fraud
    pd: np.ndarray,  # Probability of Default [N_loans]
    lgd: np.ndarray,  # Loss Given Default [N_loans]
    ead_inr: np.ndarray,  # Exposure at Default in INR [N_loans]
    tier1_buffer_inr: float,
    max_fpr: float = 0.005,
) -> SolvencyReport:
    """Enforces FPR <= 0.5% and Basel III Expected Loss <= Tier-1 Capital Buffer."""
    legit_mask = fraud_labels == 0
    fpr = float(np.mean(fraud_actions[legit_mask] != 0)) if np.any(legit_mask) else 0.0

    expected_loss = float(np.sum(pd * lgd * ead_inr))

    return SolvencyReport(
        fpr=fpr,
        expected_loss_inr=expected_loss,
        fpr_compliant=(fpr <= max_fpr),
        basel_compliant=(expected_loss <= tier1_buffer_inr),
    )
