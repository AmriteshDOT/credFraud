import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


class CalibratedFraudBaseline:
    """
    Phase I Feasible Fraud Policy pi_{F,0}:
      1. LGBMClassifier trained on D_train
      2. IsotonicRegression probability calibration on D_cal
      3. Bayes-optimal rupee expected utility with Lagrangian FPR dual (lambda_FPR*)
      4. Split Conformal Prediction abstention gate routing ambiguous states to Soft_Hold (2)
      5. Total, NaN-safe inference implementing predict_proba(states) -> [N, 4]
    """

    def __init__(
        self,
        amount_col_idx: int = 0,
        alpha_conformal: float = 0.05,
        max_fpr: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.amount_col_idx = amount_col_idx
        self.alpha_conformal = alpha_conformal
        self.max_fpr = max_fpr

        self.model = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,  # Deterministic single-thread inference for p99.9 latency stability
            verbose=-1,
        )
        self.calibrator = IsotonicRegression(
            y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip"
        )
        self.q_conformal: float = 1.0
        self.lambda_fpr: float = 0.0

    @staticmethod
    def _sanitize(states: np.ndarray) -> np.ndarray:
        """Guarantees totality on NaN/Inf stress inputs."""
        return np.nan_to_num(
            np.asarray(states, dtype=np.float64), nan=0.0, posinf=1e8, neginf=-1e8
        )

    def _expected_utility(
        self,
        p_fraud: np.ndarray,
        amounts_inr: np.ndarray,
        lam_fpr: float,
    ) -> np.ndarray:
        """
        Computes E[R_F(s, a, Y) - lam_fpr * I(a != 0) * (1 - Y)] for a in {0, 1, 2, 3}.
        Actions: 0=Approve, 1=2FA, 2=Soft_Hold, 3=Hard_Block.
        """
        p = p_fraud
        q = 1.0 - p
        amt = np.maximum(amounts_inr, 0.0)

        # Payoffs under legitimate (y=0) and fraud (y=1) outcomes in INR
        r0_legit = 0.002 * amt
        r0_fraud = -(amt + 2500.0)

        r1_legit = np.full_like(amt, -15.0)
        r1_fraud = -0.15 * (amt + 2500.0) - 15.0  # 85% fraud deterrence via 2FA

        r2_legit = np.full_like(amt, -150.0)
        r2_fraud = np.full_like(amt, -100.0)  # Ops review cost; principal saved

        r3_legit = np.full_like(amt, -500.0)  # Legitimate customer churn cost
        r3_fraud = np.zeros_like(amt)

        u0 = p * r0_fraud + q * r0_legit
        u1 = p * r1_fraud + q * (r1_legit - lam_fpr)
        u2 = p * r2_fraud + q * (r2_legit - lam_fpr)
        u3 = p * r3_fraud + q * (r3_legit - lam_fpr)

        return np.stack([u0, u1, u2, u3], axis=1)

    def _select_actions(
        self,
        p_fraud: np.ndarray,
        amounts_inr: np.ndarray,
        lam_fpr: float,
    ) -> np.ndarray:
        utils = self._expected_utility(p_fraud, amounts_inr, lam_fpr)
        actions = np.argmax(utils, axis=1)

        # Split Conformal abstention: if both y=0 and y=1 are in Gamma_alpha(s), route to Soft_Hold (2)
        in_set_fraud = (1.0 - p_fraud) <= self.q_conformal
        in_set_legit = p_fraud <= self.q_conformal
        ambiguous = in_set_fraud & in_set_legit
        actions = np.where(ambiguous, 2, actions)
        return actions

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "CalibratedFraudBaseline":
        x_tr = self._sanitize(x_train)
        x_ca = self._sanitize(x_cal)
        y_tr = np.asarray(y_train, dtype=np.int64)
        y_ca = np.asarray(y_cal, dtype=np.int64)

        # 1. Fit raw GBDT risk scorer on D_train
        self.model.fit(x_tr, y_tr)

        # 2. Fit Isotonic probability calibrator strictly on D_cal
        raw_cal = self.model.predict_proba(x_ca)[:, 1]
        self.calibrator.fit(raw_cal, y_ca)
        p_cal = self.calibrator.predict(raw_cal)

        # 3. Compute Split Conformal quantile q_{1 - alpha} on D_cal
        n_cal = len(y_ca)
        true_class_prob = np.where(y_ca == 1, p_cal, 1.0 - p_cal)
        nonconformity = 1.0 - true_class_prob
        q_level = min(np.ceil((n_cal + 1) * (1.0 - self.alpha_conformal)) / n_cal, 1.0)
        self.q_conformal = float(np.quantile(nonconformity, q_level, method="higher"))

        # 4. 1D Bisection for minimum lambda_FPR* satisfying FPR <= max_fpr on D_cal
        amounts_cal = x_ca[:, self.amount_col_idx]
        legit_mask = y_ca == 0

        lo, hi = 0.0, 1e6
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            actions = self._select_actions(p_cal, amounts_cal, lam_fpr=mid)
            fpr = float(np.mean(actions[legit_mask] != 0))
            if fpr > self.max_fpr:
                lo = mid
            else:
                hi = mid
        self.lambda_fpr = hi
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Total, NaN-safe deterministic policy returning action probabilities [N, 4]."""
        x = self._sanitize(states)
        raw = self.model.predict_proba(x)[:, 1]
        p_fraud = self.calibrator.predict(raw)
        amounts = x[:, self.amount_col_idx]

        actions = self._select_actions(p_fraud, amounts, lam_fpr=self.lambda_fpr)
        probs = np.zeros((len(x), 4), dtype=np.float64)
        probs[np.arange(len(x)), actions] = 1.0
        return probs
