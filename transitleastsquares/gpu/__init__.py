"""GPU kernels (numba-cuda, FP32) for the single transit events and their
period fold, shared by the TLS backend "events-gpu" and gp.search(device=
"gpu"). See GPU_PLAN.md (TLS-detrend) for the design.

Precision: the search kernels compute in FP32; only time differences t_i - t_c
and the transit offsets round(n P / dj) use FP64 (once per sample and per transit).
The Kalman pass, the trend and the fit's likelihood run on the device in FP64, per
segment, joined on the CPU (driver._Elements): one step costs about 1.4x FP32 on
this card, although FP64 has about 1/64 of its peak rate, because a thread's speed limits the kernels.
On the CPU, in FP64: the joins and the statistics. Results equal the CPU kernels up
to FP32 rounding and the pooling of time steps (statespace.DT_TOL), not bit for bit.

Needs numba-cuda (pip install "numba-cuda[cu13]") and an NVIDIA driver.
"""


def available():
    """True if numba-cuda is installed and a CUDA device is usable."""
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False
