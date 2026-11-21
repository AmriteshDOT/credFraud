import time
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SLAReport:
    p99_ms: float
    p999_ms: float
    violation_rate: float
    is_compliant: bool


def measure_inference_ms(fn, *args, **kwargs) -> tuple[object, float]:
    """Measures wall-clock inference latency using monotonic perf_counter."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def check_latency_sla(
    latencies_ms: np.ndarray,
    tau_sla_ms: float = 35.0,
    max_violation_prob: float = 0.001,
) -> SLAReport:
    """Enforces P(T_infer > 35ms) <= 0.001."""
    violation_rate = float(np.mean(latencies_ms > tau_sla_ms))
    return SLAReport(
        p99_ms=float(np.percentile(latencies_ms, 99.0)),
        p999_ms=float(np.percentile(latencies_ms, 99.9)),
        violation_rate=violation_rate,
        is_compliant=(violation_rate <= max_violation_prob),
    )
