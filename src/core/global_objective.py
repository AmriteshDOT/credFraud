import numpy as np
from dataclasses import dataclass


@dataclass
class EnterpriseScore:
    fraud_npv_inr: float
    credit_npv_inr: float
    total_j_inr: float
    fpr: float
    avg_daily_contacts: float
    is_feasible: bool


def evaluate_global_objective(
    fraud_rewards: np.ndarray,  # [N_sessions, T_f] discounted session rewards
    fraud_actions: np.ndarray,  # [N_sessions, T_f]
    fraud_labels: np.ndarray,  # [N_sessions, T_f]
    credit_rewards: np.ndarray,  # [N_loans, T_c] daily net recoveries (T_c = 180)
    credit_contacts: np.ndarray,  # [N_loans, T_c] binary contact indicators
    gamma_f: float = 0.90,
    gamma_c: float = 0.9997,
    max_fpr: float = 0.005,
    max_daily_contacts: float = 3.0 / 7.0,
) -> EnterpriseScore:
    """Computes J(pi_F, pi_C) = E[sum gamma_F^t R_F] + E[sum gamma_C^tau R_C]."""
    # Left Half: FraudShield expected discounted return
    t_f = np.arange(fraud_rewards.shape[1])
    fraud_npv = float(np.mean(fraud_rewards @ (gamma_f**t_f)))

    # Right Half: Credit Collections expected discounted return
    t_c = np.arange(credit_rewards.shape[1])
    credit_npv = float(np.mean(credit_rewards @ (gamma_c**t_c)))

    # Hard Constraint Checks (Feasibility precedes optimality)
    legit_mask = fraud_labels == 0
    fpr = float(np.mean(fraud_actions[legit_mask] != 0)) if np.any(legit_mask) else 0.0
    avg_contacts = float(np.mean(credit_contacts))
    is_feasible = (fpr <= max_fpr) and (avg_contacts <= max_daily_contacts)

    return EnterpriseScore(
        fraud_npv_inr=fraud_npv,
        credit_npv_inr=credit_npv,
        total_j_inr=fraud_npv + credit_npv if is_feasible else float("-inf"),
        fpr=fpr,
        avg_daily_contacts=avg_contacts,
        is_feasible=is_feasible,
    )
