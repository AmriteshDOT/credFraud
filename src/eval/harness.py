import time
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.stats import bootstrap


class FraudPolicy(Protocol):
    """Decoupled interface: any policy mapping state matrix [N, D] -> action probs [N, |A|]."""

    def predict_proba(self, states: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class OOTFraudDataset:
    """Frozen Out-of-Time evaluation measure (D_OOT)."""

    states: np.ndarray  # [N, D] point-in-time causal features
    logged_actions: np.ndarray  # [N] int in {0, 1, 2, 3}
    logged_rewards: np.ndarray  # [N] realized rupee rewards R_F
    y_true: np.ndarray  # [N] matured ground-truth labels {0, 1}
    propensities: np.ndarray  # [N] pi_b(a_i | s_i) > 0
    q_hat: np.ndarray  # [N, |A|] direct-method nuisance reward matrix


@dataclass(frozen=True, slots=True)
class EvalReport:
    dr_mean_inr: float
    dr_lcb_95_inr: float
    effective_j_inr: float
    fpr: float
    p999_latency_ms: float
    totality_passed: bool
    is_feasible: bool


class EvaluationOracle:
    """Higher-order functional O(pi) -> EvalReport."""

    def __init__(
        self,
        oot_data: OOTFraudDataset,
        stress_states: np.ndarray,  # D_stress: NaNs, zeros, extreme values
        max_fpr: float = 0.005,
        tau_sla_ms: float = 35.0,
        clip_c: float = 10.0,
        n_resamples: int = 2000,
        seed: int = 42,
    ) -> None:
        self.oot = oot_data
        self.stress_states = stress_states
        self.max_fpr = max_fpr
        self.tau_sla_ms = tau_sla_ms
        self.clip_c = clip_c
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def _check_totality(self, policy: FraudPolicy) -> bool:
        """Asserts pi(x) != bot (no crashes, no NaNs, valid probability simplex) on D_stress."""
        try:
            probs = policy.predict_proba(self.stress_states)
            if not np.all(np.isfinite(probs)):
                return False
            if np.any(probs < 0.0) or not np.allclose(
                probs.sum(axis=1), 1.0, atol=1e-5
            ):
                return False
            return True
        except Exception:
            return False

    def _measure_p999_latency(self, policy: FraudPolicy, n_trials: int = 200) -> float:
        """Measures single-event inference latency p99.9 in milliseconds."""
        sample_idx = self.rng.integers(0, len(self.oot.states), size=n_trials)
        latencies_ms = np.empty(n_trials, dtype=np.float64)
        for i, idx in enumerate(sample_idx):
            x = self.oot.states[idx : idx + 1]
            t0 = time.perf_counter()
            policy.predict_proba(x)
            latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
        return float(np.percentile(latencies_ms, 99.9))

    def evaluate(self, policy: FraudPolicy) -> EvalReport:
        # 1. Totality & Latency Constraints (Walls 1 & Edge Safety)
        totality_ok = self._check_totality(policy)
        p999_ms = self._measure_p999_latency(policy)

        # 2. Policy Action Probabilities on D_OOT
        pi_probs = policy.predict_proba(self.oot.states)  # [N, |A|]
        n = len(self.oot.logged_actions)
        row_idx = np.arange(n)

        # 3. Clipped Doubly Robust (DR) Per-Sample Value Estimates
        v_hat = np.sum(pi_probs * self.oot.q_hat, axis=1)  # sum_a pi(a|s) Q_hat(s,a)
        pi_taken = pi_probs[row_idx, self.oot.logged_actions]  # pi(a_i | s_i)
        q_taken = self.oot.q_hat[row_idx, self.oot.logged_actions]  # Q_hat(s_i, a_i)
        rho = np.minimum(
            self.clip_c, pi_taken / self.oot.propensities
        )  # Clipped IPS weight

        dr_scores = v_hat + rho * (self.oot.logged_rewards - q_taken)
        dr_mean = float(np.mean(dr_scores))

        # 4. One-Sided 95% Lower Confidence Bound via SciPy Bootstrap
        boot = bootstrap(
            (dr_scores,),
            np.mean,
            n_resamples=self.n_resamples,
            confidence_level=0.95,
            alternative="greater",  # One-sided lower bound: [low, +inf)
            method="percentile",
            rng=self.rng,
        )
        dr_lcb = float(boot.confidence_interval.low)

        # 5. Solvency Constraint: FPR <= 0.5% on legitimate transactions
        greedy_actions = np.argmax(pi_probs, axis=1)
        legit_mask = self.oot.y_true == 0
        fpr = (
            float(np.mean(greedy_actions[legit_mask] != 0))
            if np.any(legit_mask)
            else 0.0
        )

        is_feasible = (
            totality_ok and (p999_ms <= self.tau_sla_ms) and (fpr <= self.max_fpr)
        )
        effective_j = dr_lcb if is_feasible else float("-inf")

        return EvalReport(
            dr_mean_inr=dr_mean,
            dr_lcb_95_inr=dr_lcb,
            effective_j_inr=effective_j,
            fpr=fpr,
            p999_latency_ms=p999_ms,
            totality_passed=totality_ok,
            is_feasible=is_feasible,
        )
