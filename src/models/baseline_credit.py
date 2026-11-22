import numpy as np
from scipy.optimize import linprog


class CMDPCreditBaseline:
    """
    Phase I Feasible Credit Collection Policy pi_{C,0}.
    Solves the RBI-constrained discounted MDP J_C(pi_C) exactly via its
    occupancy-measure Linear Program (Altman, 1999).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma_c: float = (1.0 + 0.12) ** (-1.0 / 365.0),  # Daily discount (12% WACC)
        c_max_rbi: float = 15.0,  # Max discounted RBI contacts
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma_c = gamma_c
        self.c_max_rbi = c_max_rbi
        self.policy_matrix: np.ndarray = np.zeros(
            (num_states, num_actions), dtype=np.float64
        )
        self.policy_matrix[:, 0] = 1.0  # Safe default: Action 0 (No_Contact)
        self.optimal_npv_inr: float = 0.0

    def fit(
        self,
        P_hat: np.ndarray,  # Transition kernel [S, A, S'] where P_hat[s, a, s'] = P(s'|s,a)
        R_c: np.ndarray,  # Reward matrix [S, A] in INR
        C_rbi: np.ndarray,  # RBI contact cost matrix [S, A] in {0, 1}
        mu_0: np.ndarray,  # Initial borrower state distribution [S]
    ) -> "CMDPCreditBaseline":
        S, A = self.num_states, self.num_actions
        num_vars = S * A

        # 1. Objective: maximize sum_{s,a} rho(s,a) R_c(s,a) <=> minimize -R_c^T rho
        c_obj = -R_c.reshape(num_vars).astype(np.float64)

        # 2. Bellman flow conservation:
        #    sum_a rho(s', a) - gamma_c * sum_{s,a} rho(s, a) P_hat(s'|s,a) = mu_0(s')
        A_eq = np.zeros((S, num_vars), dtype=np.float64)
        for s_next in range(S):
            for s in range(S):
                for a in range(A):
                    idx = s * A + a
                    A_eq[s_next, idx] = (
                        1.0 if s == s_next else 0.0
                    ) - self.gamma_c * P_hat[s, a, s_next]
        b_eq = np.asarray(mu_0, dtype=np.float64)

        # 3. RBI statutory contact cap: sum_{s,a} rho(s,a) C_rbi(s,a) <= c_max_rbi
        A_ub = C_rbi.reshape(1, num_vars).astype(np.float64)
        b_ub = np.array([self.c_max_rbi], dtype=np.float64)

        # 4. Solve exact LP with rho(s, a) >= 0
        res = linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Credit CMDP LP infeasible: {res.message}")

        rho_star = res.x.reshape(S, A)
        self.optimal_npv_inr = float(-res.fun)

        # 5. Extract stationary randomized policy pi_{C,0}(a | s)
        state_occ = rho_star.sum(axis=1, keepdims=True)
        policy = np.zeros((S, A), dtype=np.float64)
        visited = state_occ[:, 0] > 1e-12
        policy[visited] = rho_star[visited] / state_occ[visited]
        policy[~visited, 0] = 1.0  # Totality fallback -> No_Contact

        self.policy_matrix = policy
        return self

    def predict_proba(self, state_indices: np.ndarray) -> np.ndarray:
        """Total, bounds-safe policy returning action probabilities [N, A_C]."""
        idx = np.clip(np.asarray(state_indices, dtype=np.int64), 0, self.num_states - 1)
        return self.policy_matrix[idx]
