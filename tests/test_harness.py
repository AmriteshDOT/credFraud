import numpy as np
from src.eval.harness import EvaluationOracle, OOTFraudDataset


class ConstantPolicy:
    """Analytical dummy policy placing probability 1.0 on a single fixed action."""

    def __init__(self, action_idx: int, num_actions: int = 4) -> None:
        self.action_idx = action_idx
        self.num_actions = num_actions

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(states), self.num_actions), dtype=np.float64)
        probs[:, self.action_idx] = 1.0
        return probs


def test_harness_rejects_infeasible_block_all_and_accepts_approve_all():
    rng = np.random.default_rng(0)
    n = 500
    oot = OOTFraudDataset(
        states=rng.normal(size=(n, 5)),
        logged_actions=np.zeros(n, dtype=np.int64),
        logged_rewards=np.full(n, 15.0),
        y_true=np.zeros(n, dtype=np.int64),  # All legit
        propensities=np.ones(n, dtype=np.float64),
        q_hat=np.full((n, 4), 15.0),
    )
    # Stress set containing cold-start NaNs and extreme values
    stress = np.array([[np.nan, 0.0, 1e9, -1e9, 0.0]])

    oracle = EvaluationOracle(oot_data=oot, stress_states=stress, n_resamples=200)

    # 1. Block-All Policy: FPR = 1.0 > 0.005 -> MUST be infeasible (-inf)
    rep_block = oracle.evaluate(ConstantPolicy(action_idx=3))
    assert rep_block.fpr == 1.0
    assert not rep_block.is_feasible
    assert rep_block.effective_j_inr == float("-inf")

    # 2. Approve-All Policy: FPR = 0.0, exact DR mean = 15.0 INR -> MUST pass
    rep_approve = oracle.evaluate(ConstantPolicy(action_idx=0))
    assert rep_approve.fpr == 0.0
    assert rep_approve.is_feasible
    assert np.isclose(rep_approve.dr_mean_inr, 15.0)
