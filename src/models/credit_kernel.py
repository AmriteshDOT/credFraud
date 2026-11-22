import numpy as np
from scipy.optimize import minimize


class CoxTransitionKernel:
    """
    Block 3 Optimization: Competing-Risks Cox Proportional Hazards estimator
    producing a right-censored 1-day transition kernel P_hat_{C,1}[S, A, S']
    for CMDPCreditBaseline.
    """

    def __init__(
        self, num_states: int = 3, num_actions: int = 3, l2_reg: float = 1e-2
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.l2_reg = l2_reg
        self.beta_cure: np.ndarray | None = None
        self.beta_default: np.ndarray | None = None
        self.h0_cure_1d: float = 0.0
        self.h0_default_1d: float = 0.0

    def _encode_sa(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """One-hot encodes (s, a) -> psi(s, a) in R^(S + A)."""
        n = len(states)
        psi = np.zeros((n, self.num_states + self.num_actions), dtype=np.float64)
        psi[np.arange(n), states] = 1.0
        psi[np.arange(n), self.num_states + actions] = 1.0
        return psi

    def _fit_cause_specific(
        self,
        psi: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Minimizes regularized negative log partial likelihood + Breslow 1-day hazard."""
        # Sort by duration descending so risk set R(t_i) = prefix [0..i]
        order = np.argsort(-durations, kind="mergesort")
        x = psi[order]
        t = durations[order]
        e = events[order].astype(np.float64)

        def neg_log_partial_lik(beta: np.ndarray) -> float:
            eta = np.clip(x @ beta, -20.0, 20.0)
            exp_eta = np.exp(eta)
            cum_risk = np.cumsum(exp_eta)
            ll = np.sum(e * (eta - np.log(np.maximum(cum_risk, 1e-12))))
            reg = 0.5 * self.l2_reg * float(np.dot(beta, beta))
            return float(-ll + reg)

        res = minimize(
            neg_log_partial_lik,
            x0=np.zeros(x.shape[1], dtype=np.float64),
            method="BFGS",
        )
        beta_hat = res.x

        # Breslow average 1-day baseline hazard
        exp_eta = np.exp(np.clip(x @ beta_hat, -20.0, 20.0))
        cum_risk = np.cumsum(exp_eta)
        dh0 = np.sum(e / np.maximum(cum_risk, 1e-12))
        t_obs = max(float(np.max(t)), 1.0)
        return beta_hat, float(dh0 / t_obs)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        durations_days: np.ndarray,
        event_types: np.ndarray,  # 0=censored, 1=cure (to state 0), 2=default (to state S-1)
    ) -> "CoxTransitionKernel":
        psi = self._encode_sa(states, actions)
        self.beta_cure, self.h0_cure_1d = self._fit_cause_specific(
            psi, durations_days, (event_types == 1)
        )
        self.beta_default, self.h0_default_1d = self._fit_cause_specific(
            psi, durations_days, (event_types == 2)
        )
        return self

    def build_transition_matrix(self) -> np.ndarray:
        """Constructs stochastic kernel P_hat_{C,1}[S, A, S'] for CMDPCreditBaseline.fit()."""
        S, A = self.num_states, self.num_actions
        P_hat = np.zeros((S, A, S), dtype=np.float64)

        for s in range(S):
            for a in range(A):
                if s == S - 1:
                    # Default is absorbing
                    P_hat[s, a, S - 1] = 1.0
                    continue

                psi = self._encode_sa(np.array([s]), np.array([a]))[0]
                h_cure = self.h0_cure_1d * np.exp(
                    np.clip(psi @ self.beta_cure, -20.0, 20.0)
                )
                h_def = self.h0_default_1d * np.exp(
                    np.clip(psi @ self.beta_default, -20.0, 20.0)
                )

                p_cure = 1.0 - np.exp(-h_cure)
                p_def = 1.0 - np.exp(-h_def)

                # Normalize if competing hazards exceed 1.0
                total_exit = p_cure + p_def
                if total_exit > 1.0:
                    p_cure /= total_exit
                    p_def /= total_exit
                    p_stay = 0.0
                else:
                    p_stay = 1.0 - total_exit

                P_hat[s, a, 0] += p_cure
                P_hat[s, a, S - 1] += p_def
                P_hat[s, a, s] += p_stay

        return P_hat
