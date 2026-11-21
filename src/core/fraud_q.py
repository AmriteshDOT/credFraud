import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudQNetwork(nn.Module):
    """Approximates Q_phi(s_t^F, a_t^F) for discrete fraud actions |A_F| = 4."""

    def __init__(self, state_dim: int, num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # Output shape: [Batch, 4]


class FraudCQLTrainer:
    """Solves min_phi L_CQL(phi) over maturity-censored offline replay buffer D."""

    def __init__(
        self,
        state_dim: int,
        gamma_f: float = 0.90,
        alpha_cql: float = 1.0,
        lr: float = 3e-4,
    ):
        self.gamma_f = gamma_f
        self.alpha_cql = alpha_cql
        self.q_net = FraudQNetwork(state_dim)
        self.target_net = FraudQNetwork(state_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)

    def train_step(self, s, a, r, s_next, done) -> float:
        q_all = self.q_net(s)  # [B, 4]
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # Q(s, a_data)

        with torch.no_grad():
            next_max_q = self.target_net(s_next).max(dim=1).values
            td_target = r + self.gamma_f * (1.0 - done) * next_max_q

        bellman_loss = 0.5 * F.mse_loss(q_taken, td_target)
        # CQL conservative penalty: pushes down OOD actions, pushes up dataset actions
        cql_penalty = self.alpha_cql * (torch.logsumexp(q_all, dim=1) - q_taken).mean()
        loss = bellman_loss + cql_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state).argmax(dim=-1)
