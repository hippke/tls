"""GPU kernels (numba-cuda, FP32) for the single transit events and their
period fold, shared by the TLS backend "events-gpu" and gp.search(device=
"gpu"). See GPU_PLAN.md (TLS-detrend) for the design.

Precision: the kernels compute in FP32 (consumer cards run FP64 at ~1/64 of
the FP32 rate); only time differences t_i - t_c and the transit offsets
round(n P / dj) use FP64 (once per sample and per transit). On the device too:
the covariance recursion of the Kalman pass (as the change from FP64 key
covariances, driver.kalman_gpu) and the tail recursion. On the CPU, in FP64: the
coarse key pass, the state recursion, the GP fit, statistics. Results equal the
CPU kernels up to FP32 rounding, not bit for bit.

Needs numba-cuda (pip install "numba-cuda[cu13]") and an NVIDIA driver.
"""


def available():
    """True if numba-cuda is installed and a CUDA device is usable."""
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False
