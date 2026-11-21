import numpy as np


# Action indices for FraudShield: 0=Approve, 1=StepUp_2FA, 2=Soft_Hold, 3=Hard_Block
def compute_fraud_reward(
    action: np.ndarray,  # shape [N]
    y_true: np.ndarray,  # shape [N], 1=fraud, 0=legit
    amount_inr: np.ndarray,  # shape [N]
    margin_m: float = 0.015,
    cb_fee: float = 500.0,
    reg_penalty: float = 2000.0,
) -> np.ndarray:
    """R_F(s_t^F, a_t^F, y_t): Exact asymmetric rupee reward per transaction."""
    approve = (action == 0).astype(np.float64)
    intervene = (action != 0).astype(np.float64)

    # Operational & friction costs by action: [Approve, 2FA, SoftHold_Agent, HardBlock]
    c_op = np.array([0.0, 1.5, 45.0, 2.0])[action]
    c_fric = np.array([0.0, 15.0, 120.0, 800.0])[action]

    gain_legit = approve * (1.0 - y_true) * (margin_m * amount_inr)
    loss_fraud = approve * y_true * (amount_inr + cb_fee + reg_penalty)
    loss_friction = intervene * (1.0 - y_true) * c_fric

    return gain_legit - loss_fraud - loss_friction - c_op
