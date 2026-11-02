"""transitleastsquares.gp must run without SciPy (a test-only dependency of
TLS): fits, flare clipping (smoother) and the search, in a subprocess where importing scipy fails, with a fresh numba
cache (numba takes BLAS/LAPACK for @ and np.linalg from SciPy, so compiled
kernels must use explicit loops)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import sys
class Block:
    def find_spec(self, name, path, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked")
sys.meta_path.insert(0, Block())
import numpy as np
from transitleastsquares import gp
from gp_cases import synthetic
t, f, e = synthetic(days=12.0)
gp.search(t, f, e, baseline="spline", period_min=1.0, period_max=4.0)
r = gp.search(t, f, e, rotation_period=2.3, n_starts=1, period_min=1.0, period_max=4.0, bayes_factor=True)
assert not any(m == "scipy" or m.startswith("scipy.") for m in sys.modules)
print("OK", round(r["period"], 4))
"""


@pytest.mark.slow
def test_runs_without_scipy(tmp_path):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba"),
               PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "tests"), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("OK")
