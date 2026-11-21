import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from src.core.credit_v import CreditCMDPSolver, CMDPSolution


class CoxCreditBaseline:
    """Estimates right-censored daily transition probabilities via Cox PH and solves CMDP."""

    def __init__(self, penalizer: float = 0.01) -> None:
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.solver = CreditCMDPSolver(gamma_c=0.9997, contact_cap=3.0 / 7.0)

    def fit_hazard(
        self,
        panel_df: pd.DataFrame,
        duration_col: str = "day_tau",
        event_col: str = "is_event_observed",
    ) -> "CoxCreditBaseline":
        """Fits Cox partial likelihood over censored borrower histories."""
        self.cph.fit(panel_df, duration_col=duration_col, event_col=event_col)
        return self

    def predict_daily_event_prob(
        self, state_action_df: pd.DataFrame, day_tau: int
    ) -> np.ndarray:
        """Computes P(event in [tau, tau+1) | s, a) = 1 - S(tau+1) / S(tau)."""
        surv = self.cph.predict_survival_function(
            state_action_df, times=[day_tau, day_tau + 1]
        )
        s_t = np.clip(surv.loc[day_tau].to_numpy(), 1e-8, 1.0)
        s_next = np.clip(surv.loc[day_tau + 1].to_numpy(), 0.0, s_t)
        return 1.0 - (s_next / s_t)

    def solve_baseline_policy(
        self,
        P_hat: np.ndarray,  # [S, A, S] built from predict_daily_event_prob
        R_hat: np.ndarray,  # [S, A] expected net rupee recovery
        C_mat: np.ndarray,  # [S, A] binary RBI contact indicator
    ) -> CMDPSolution:
        return self.solver.solve(P=P_hat, R=R_hat, C=C_mat)
