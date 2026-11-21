import numpy as np
from dataclasses import dataclass


@dataclass
class CMDPSolution:
    policy: np.ndarray  # Optimal action pi_C^*(s) for every delinquency state
    value_fn: np.ndarray  # Optimal discounted NPV V^*(s)
    dual_lambda: float  # Converged Lagrange multiplier for RBI contact cap


class CreditCMDPSolver:
    """Solves max_{pi_C} V(s) - lambda * C(s) via Primal-Dual Value Iteration."""

    def __init__(self, gamma_c: float = 0.9997, contact_cap: float = 3.0 / 7.0):
        self.gamma_c = gamma_c
        self.contact_cap = contact_cap  # Max avg daily contacts (3 per 7 days)

    def solve(
        self,
        P: np.ndarray,  # Transition tensor P(s'|s,a), shape [S, A, S]
        R: np.ndarray,  # Expected net rupee recovery R_C(s,a), shape [S, A]
        C: np.ndarray,  # Regulatory contact cost c(s,a) in {0, 1}, shape [S, A]
        dual_lr: float = 0.05,
        outer_iters: int = 100,
        inner_tol: float = 1e-6,
    ) -> CMDPSolution:
        num_states, num_actions, _ = P.shape
        lam = 0.0
        V = np.zeros(num_states, dtype=np.float64)
        policy = np.zeros(num_states, dtype=np.int64)

        for _ in range(outer_iters):
            # 1. Inner Primal: Bellman Contraction Mapping
            for _ in range(500):
                Q = (R - lam * C) + self.gamma_c * np.einsum("ijk,k->ij", P, V)
                V_new = Q.max(axis=1)
                if np.max(np.abs(V_new - V)) < inner_tol:
                    V = V_new
                    break
                V = V_new

            policy = Q.argmax(axis=1)

            # 2. Outer Dual Ascent: Enforce E[C] <= contact_cap
            expected_contacts = C[np.arange(num_states), policy].mean()
            lam = max(0.0, lam + dual_lr * (expected_contacts - self.contact_cap))

        return CMDPSolution(policy=policy, value_fn=V, dual_lambda=lam)
