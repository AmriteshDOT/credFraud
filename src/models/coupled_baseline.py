import numpy as np
from src.models.baseline_fraud import CalibratedFraudBaseline
from src.models.baseline_credit import CMDPCreditBaseline


class CoupledEnterpriseBaseline:
    """
    Unified Phase I Feasible Solution pi_0 = (pi_{F,0}, pi_{C,0})
    maximizing J_0(pi_F, rho) = J_F^(0)(pi_F) + J_C^(0)(rho) subject to V_C(pi_0) = 1.
    """

    def __init__(
        self,
        fraud_policy: CalibratedFraudBaseline,
        credit_policy: CMDPCreditBaseline,
    ) -> None:
        self.fraud_policy = fraud_policy
        self.credit_policy = credit_policy

    def decide_fraud(self, x_fraud: np.ndarray) -> np.ndarray:
        """Real-time payment interdiction (ms regime)."""
        return self.fraud_policy.predict_proba(x_fraud)

    def decide_credit(self, credit_state_ids: np.ndarray) -> np.ndarray:
        """Daily multi-horizon credit collection decision (day regime)."""
        return self.credit_policy.predict_proba(credit_state_ids)
