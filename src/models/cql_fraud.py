from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def build_transitions(
    payer_ids: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs Markov transitions (s_t, a_t, r_t, s_{t+1}, d_t) from a
    chronologically sorted dataset grouped by payer_id.
    """
    order = np.argsort(payer_ids, kind="mergesort")
    p_sorted = payer_ids[order]
    s_sorted = states[order]
    a_sorted = actions[order]
    r_sorted = rewards[order]

    dones = np.ones(len(p_sorted), dtype=np.float32)
    dones[:-1] = (p_sorted[:-1] != p_sorted[1:]).astype(np.float32)

    s_next = np.zeros_like(s_sorted, dtype=np.float32)
    s_next[:-1] = s_sorted[1:]
    s_next[dones == 1.0] = 0.0

    return (
        s_sorted.astype(np.float32),
        a_sorted.astype(np.int64),
        r_sorted.astype(np.float32),
        s_next,
        dones,
    )


class QNetwork(nn.Module):
    """Compact 2-layer MLP Q_theta(s) -> R^4 for <1ms single-thread CPU inference."""

    def __init__(
        self, state_dim: int, num_actions: int = 4, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CQLFraudPolicy:
    """
    Block 2 Sequential Fraud Policy pi_{F,1}:
      1. Discrete Conservative Q-Learning (CQL(H)) on D_train transitions
      2. Polyak target network updates
      3. Lagrangian FPR dual tuning (lambda_FPR*) on D_cal
      4. Total, NaN-safe, <35ms inference implementing predict_proba(states) -> [N, 4]
    """

    def __init__(
        self,
        state_dim: int = 11,
        num_actions: int = 4,
        gamma: float = 0.90,
        alpha_cql: float = 1.0,
        tau: float = 0.005,
        lr: float = 3e-4,
        reward_scale: float = 1000.0,
        max_fpr: float = 0.005,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha_cql = alpha_cql
        self.tau = tau
        self.reward_scale = reward_scale
        self.max_fpr = max_fpr

        self.q_net = QNetwork(state_dim, num_actions)
        self.target_net = deepcopy(self.q_net)
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.q_net.parameters(), lr=lr, weight_decay=1e-4
        )

        self.mean_: np.ndarray = np.zeros(state_dim, dtype=np.float32)
        self.std_: np.ndarray = np.ones(state_dim, dtype=np.float32)
        self.lambda_fpr: float = 0.0

    def _normalize(self, states: np.ndarray) -> np.ndarray:
        """NaN-safe z-score normalization clipped to [-10, 10] for totality."""
        clean = np.nan_to_num(states, nan=0.0, posinf=1e8, neginf=-1e8).astype(
            np.float32
        )
        z = (clean - self.mean_) / (self.std_ + 1e-6)
        return np.clip(z, -10.0, 10.0)

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.q_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _predict_q_values(self, states: np.ndarray) -> np.ndarray:
        self.q_net.eval()
        x_norm = self._normalize(states)
        with torch.no_grad():
            return self.q_net(torch.from_numpy(x_norm)).numpy()

    def _tune_fpr_dual(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Binary search on D_cal for minimum lambda_FPR* satisfying FPR <= max_fpr."""
        q_cal = self._predict_q_values(x_cal)
        legit_mask = y_cal == 0
        penalty_mask = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)

        lo, hi = 0.0, 500.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            actions = np.argmax(q_cal - mid * penalty_mask, axis=1)
            fpr = float(np.mean(actions[legit_mask] != 0))
            if fpr > self.max_fpr:
                lo = mid
            else:
                hi = mid
        self.lambda_fpr = hi

    def fit(
        self,
        payer_ids_train: np.ndarray,
        x_train: np.ndarray,
        actions_train: np.ndarray,
        rewards_train: np.ndarray,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
        epochs: int = 20,
        batch_size: int = 512,
    ) -> "CQLFraudPolicy":
        # 1. Fit standardization statistics strictly on D_train
        clean_train = np.nan_to_num(x_train, nan=0.0, posinf=1e8, neginf=-1e8).astype(
            np.float32
        )
        self.mean_ = clean_train.mean(axis=0)
        self.std_ = clean_train.std(axis=0)

        # 2. Build Markov transitions (s, a, r, s', d)
        s, a, r, s_next, d = build_transitions(
            payer_ids=payer_ids_train,
            states=self._normalize(x_train),
            actions=actions_train,
            rewards=rewards_train / self.reward_scale,
        )

        dataset = TensorDataset(
            torch.from_numpy(s),
            torch.from_numpy(a),
            torch.from_numpy(r),
            torch.from_numpy(s_next),
            torch.from_numpy(d),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Minimize Discrete CQL(H) + Bellman TD Loss
        self.q_net.train()
        for _ in range(epochs):
            for sb, ab, rb, s_next_b, db in loader:
                q_all = self.q_net(sb)  # [B, 4]
                q_taken = q_all.gather(1, ab.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    next_q_max = self.target_net(s_next_b).max(dim=1).values
                    y_target = rb + self.gamma * (1.0 - db) * next_q_max

                cql_loss = (torch.logsumexp(q_all, dim=1) - q_taken).mean()
                td_loss = 0.5 * F.mse_loss(q_taken, y_target)
                loss = self.alpha_cql * cql_loss + td_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                self._polyak_update()

        # 4. Tune Lagrangian FPR dual strictly on D_cal
        self._tune_fpr_dual(x_cal, y_cal)
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Total, NaN-safe policy returning action probabilities [N, 4] for EvaluationOracle."""
        q_vals = self._predict_q_values(states)
        penalty_mask = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        actions = np.argmax(q_vals - self.lambda_fpr * penalty_mask, axis=1)

        probs = np.zeros((len(states), self.num_actions), dtype=np.float64)
        probs[np.arange(len(states)), actions] = 1.0
        return probs
