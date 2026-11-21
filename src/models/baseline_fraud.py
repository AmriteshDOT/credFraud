import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression


class CalibratedFraudBaseline:
    """
    Baseline policy pi_{F,0}:
      1. LightGBM (f_theta) on D_train
      2. Isotonic Calibration (p_theta) on D_cal
      3. Split Conformal Abstention (q_{1-alpha}) on D_cal
      4. Bayes-optimal cost-sensitive action + Lagrangian FPR penalty
    """

    def __init__(
        self,
        amount_col_idx: int = 0,
        alpha_conformal: float = 0.05,
        max_fpr: float = 0.005,
        margin_m: float = 0.015,
        cb_fee: float = 500.0,
        reg_penalty: float = 2000.0,
    ) -> None:
        self.amount_col_idx = amount_col_idx
        self.alpha = alpha_conformal
        self.max_fpr = max_fpr
        self.margin_m = margin_m
        self.cb_fee = cb_fee
        self.reg_penalty = reg_penalty

        # Operational & friction costs: [0=Approve, 1=2FA, 2=SoftHold, 3=HardBlock]
        self.c_op = np.array([0.0, 1.5, 45.0, 2.0], dtype=np.float64)
        self.c_fric = np.array([0.0, 15.0, 120.0, 800.0], dtype=np.float64)

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50_000,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.q_conformal: float = 1.0
        self.lambda_fpr: float = 0.0

    def _expected_utility(
        self, p_fraud: np.ndarray, amounts: np.ndarray, lam_fpr: float
    ) -> np.ndarray:
        """Computes E[R_F(s, a, y)] - lam_fpr * I(a != 0) * (1 - p) for all 4 actions."""
        p_legit = 1.0 - p_fraud
        n = len(p_fraud)
        eu = np.empty((n, 4), dtype=np.float64)

        # Action 0: Approve
        eu[:, 0] = p_legit * (self.margin_m * amounts) - p_fraud * (
            amounts + self.cb_fee + self.reg_penalty
        )

        # Actions 1, 2, 3: Intervene (2FA, Soft_Hold, Hard_Block)
        for a in (1, 2, 3):
            eu[:, a] = -p_legit * (self.c_fric[a] + lam_fpr) - self.c_op[a]

        return eu

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "CalibratedFraudBaseline":
        # 1. Fit GBDT on D_train
        self.model.fit(x_train, y_train)

        # 2. Fit Isotonic Calibrator on D_cal
        raw_cal = self.model.predict_proba(x_cal)[:, 1]
        self.calibrator.fit(raw_cal, y_cal)
        p_cal = self.calibrator.predict(raw_cal)

        # 3. Compute Split Conformal Quantile q_{1-alpha} on D_cal
        n_cal = len(y_cal)
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        level = min(1.0, np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal)
        self.q_conformal = float(np.quantile(scores, level, method="higher"))

        # 4. Tune lambda_fpr via binary search on D_cal to satisfy FPR <= max_fpr
        amounts_cal = np.clip(
            np.nan_to_num(x_cal[:, self.amount_col_idx], nan=0.0), 0.0, 1e9
        )
        legit_mask = y_cal == 0
        lo, hi = 0.0, 1e6
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            actions = np.argmax(self._expected_utility(p_cal, amounts_cal, mid), axis=1)
            fpr = float(np.mean(actions[legit_mask] != 0))
            if fpr > self.max_fpr:
                lo = mid
            else:
                hi = mid
        self.lambda_fpr = hi
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Total, NaN-safe policy returning action probabilities [N, 4] for EvaluationOracle."""
        # LightGBM handles NaN features natively; sanitize amount column for rupee math
        raw_probs = self.model.predict_proba(states)[:, 1]
        p_fraud = self.calibrator.predict(raw_probs)
        amounts = np.clip(
            np.nan_to_num(states[:, self.amount_col_idx], nan=0.0), 0.0, 1e9
        )

        # Bayes-optimal action under calibrated probabilities & tuned FPR penalty
        eu = self._expected_utility(p_fraud, amounts, self.lambda_fpr)
        actions = np.argmax(eu, axis=1)

        # Conformal Abstention: if both y=0 and y=1 are plausible (|Gamma_alpha| == 2), route to Soft_Hold (2)
        in_set_1 = (1.0 - p_fraud) <= self.q_conformal
        in_set_0 = p_fraud <= self.q_conformal
        ambiguous = in_set_1 & in_set_0
        actions = np.where(ambiguous, 2, actions)

        # Return valid probability simplex [N, 4]
        probs = np.zeros((len(states), 4), dtype=np.float64)
        probs[np.arange(len(states)), actions] = 1.0
        return probs
