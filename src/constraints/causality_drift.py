import numpy as np


def assert_point_in_time_causality(
    tx_timestamps_ms: np.ndarray,
    feature_snapshot_ts_ms: np.ndarray,
    label_maturity_ts_ms: np.ndarray,
    train_cutoff_ts_ms: int,
) -> None:
    """Hard assertion enforcing s_t^F = F(H_{(-inf, t - eps]}) and label maturity."""
    if np.any(feature_snapshot_ts_ms >= tx_timestamps_ms):
        leaked = int(np.sum(feature_snapshot_ts_ms >= tx_timestamps_ms))
        raise AssertionError(
            f"Point-in-Time Leakage Detected: {leaked} rows used future features"
        )

    if np.any(label_maturity_ts_ms > train_cutoff_ts_ms):
        immature = int(np.sum(label_maturity_ts_ms > train_cutoff_ts_ms))
        raise AssertionError(
            f"Censoring Violation: {immature} immature chargeback labels in train split"
        )


def compute_psi(
    expected_train: np.ndarray,
    actual_live: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-6,
) -> tuple[float, bool]:
    """Computes PSI = sum((P_b - Q_b) * ln(P_b / Q_b)) and checks PSI <= 0.20."""
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(expected_train, quantiles)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    q_counts, _ = np.histogram(expected_train, bins=bin_edges)
    p_counts, _ = np.histogram(actual_live, bins=bin_edges)

    q_dist = np.clip(q_counts / max(len(expected_train), 1), eps, 1.0)
    p_dist = np.clip(p_counts / max(len(actual_live), 1), eps, 1.0)

    psi = float(np.sum((p_dist - q_dist) * np.log(p_dist / q_dist)))
    return psi, (psi <= 0.20)
