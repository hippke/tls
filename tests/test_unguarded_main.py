"""Regression test for #104: a script without ``if __name__ == "__main__":``
must not hang with the "spawn" (macOS, Windows) or "forkserver" (Linux from
Python 3.14) start methods."""

import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(
    """
    import numpy
    from transitleastsquares import transitleastsquares

    t = numpy.arange(0, 30, 0.02)
    y = 1 + numpy.random.default_rng(1).normal(0, 1e-4, t.size)
    y[(t % 3.3) < 0.1] -= 1e-3
    results = transitleastsquares(t, y).power(
        period_min=2, period_max=5, use_threads=2, show_progress_bar=False
    )
    print("PERIOD", round(results.period, 2))
    """
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("persistent", ["1", "0"])
@pytest.mark.parametrize(
    "method",
    [m for m in ("spawn", "forkserver") if m in multiprocessing.get_all_start_methods()],
)
def test_unguarded_script_does_not_hang(tmp_path, method, persistent):
    script = tmp_path / "unguarded.py"
    script.write_text(SCRIPT)
    env = dict(
        os.environ,
        TLS_MP_START_METHOD=method,
        TLS_PERSISTENT_POOL=persistent,
        PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]),
    )
    kwargs = {} if sys.platform == "win32" else dict(start_new_session=True)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )
    try:
        out, err = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # the workers too
        proc.communicate()
        pytest.fail(f"power() hangs for an unguarded script with {method!r}")
    assert proc.returncode == 0, err
    assert "PERIOD 3.3" in out
