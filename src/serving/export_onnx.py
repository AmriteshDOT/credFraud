import time
import numpy as np
import torch
import onnxruntime as ort
from src.models.cql_fraud import CQLFraudPolicy


def export_cql_to_onnx(
    policy: CQLFraudPolicy, output_path: str = "artifacts/cql_fraud.onnx"
) -> None:
    """Exports trained PyTorch QNetwork to static ONNX graph for sub-millisecond serving."""
    policy.q_net.eval()
    dummy_input = torch.zeros((1, policy.state_dim), dtype=torch.float32)
    torch.onnx.export(
        policy.q_net,
        dummy_input,
        output_path,
        input_names=["state_norm"],
        output_names=["q_values"],
        dynamic_axes={"state_norm": {0: "batch_size"}, "q_values": {0: "batch_size"}},
        opset_version=17,
    )


class ONNXFraudRuntime:
    """Production <35ms inference wrapper with Adaptive Conformal Inference (ACI)."""

    def __init__(
        self,
        onnx_path: str,
        mean: np.ndarray,
        std: np.ndarray,
        lambda_fpr: float,
        alpha_target: float = 0.05,
        eta_aci: float = 0.005,
    ) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.penalty_mask = np.array(
            [0.0, lambda_fpr, lambda_fpr, lambda_fpr], dtype=np.float32
        )
        self.alpha_target = alpha_target
        self.alpha_t = alpha_target
        self.eta_aci = eta_aci

    def predict_action(self, state_vec: np.ndarray) -> int:
        clean = np.nan_to_num(state_vec, nan=0.0, posinf=1e8, neginf=-1e8).astype(
            np.float32
        )
        z = np.clip((clean - self.mean) / (self.std + 1e-6), -10.0, 10.0)
        q_vals = self.session.run(["q_values"], {"state_norm": z})[0]
        return int(np.argmax(q_vals[0] - self.penalty_mask))

    def update_aci(self, miscovered: bool) -> float:
        """Gibbs & Candes (2021) online update: alpha_{t+1} = clip(alpha_t + eta*(alpha* - err_t))."""
        err_t = 1.0 if miscovered else 0.0
        self.alpha_t = float(
            np.clip(self.alpha_t + self.eta_aci * (self.alpha_target - err_t), 0.0, 1.0)
        )
        return self.alpha_t

    def benchmark_p999_ms(self, sample_state: np.ndarray, n_runs: int = 2000) -> float:
        latencies_ms = np.empty(n_runs, dtype=np.float64)
        for i in range(n_runs):
            t0 = time.perf_counter_ns()
            self.predict_action(sample_state)
            latencies_ms[i] = (time.perf_counter_ns() - t0) / 1e6
        return float(np.percentile(latencies_ms, 99.9))
