import numpy as np


def sample_bounded_2hop_subgraph(
    adj_list: dict[str, list[str]],
    seed_account: str,
    s1_cap: int = 15,
    s2_cap: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, list[str]]:
    """
    Degree-bounded neighbor sampling (GraphSAGE-style) preventing O(k^2)
    super-node explosions on merchant VPAs or mule hubs.
    """
    rng = rng or np.random.default_rng(42)

    def _cap_neighbors(node: str, cap: int) -> list[str]:
        nbrs = adj_list.get(node, [])
        if len(nbrs) <= cap:
            return nbrs
        idx = rng.choice(len(nbrs), size=cap, replace=False)
        return [nbrs[i] for i in idx]

    hop1 = _cap_neighbors(seed_account, s1_cap)
    subgraph = {seed_account: hop1}
    for n1 in hop1:
        subgraph[n1] = _cap_neighbors(n1, s2_cap)
    return subgraph
