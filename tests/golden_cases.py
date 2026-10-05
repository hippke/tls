"""Characterization ("golden") cases: a broad set of fast TLS runs whose complete
results are frozen in tests/golden/<GOLDEN_SET>/*.npz. Any refactoring must
reproduce them (tests/test_golden.py); any intended numerical change must
regenerate them (python tests/golden_cases.py --write) and be documented.
"""

import os
import sys
import warnings

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
# Golden sets:
#   v1 = TLS 1.33 + bug fixes, batman templates. Reproduced by the refactored
#        code with TLS_GOLDEN_SET=v1 TLS_TRANSIT_MODEL=batman.
#   v2 = own transit model (transit_model.py) instead of batman. Default.
GOLDEN_SET = os.environ.get("TLS_GOLDEN_SET", "v2")
GOLDEN_DIR = os.path.join(HERE, "golden", GOLDEN_SET)
DATA = os.path.join(HERE, "data")


def trapezoid(t, period, t0, duration, depth, ingress_frac=0.15):
    """Symmetric trapezoid transit (no external dependency)."""
    ph = numpy.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
    half = 0.5 * duration
    ing = ingress_frac * duration
    y = numpy.ones_like(t)
    full = ph <= half - ing
    slope = (ph > half - ing) & (ph < half)
    y[full] -= depth
    y[slope] -= depth * (half - ph[slope]) / ing
    return y


def load_repo(name):
    import scipy.signal

    d = numpy.genfromtxt(
        os.path.join(DATA, name), delimiter=",", dtype="f8, f8", names=["t", "y"]
    )
    return d["t"], d["y"] / scipy.signal.medfilt(d["y"], 25)


def synthetic_earth(nan_gap=False, with_dy=False):
    # Same light curve as test_synthetic.py / test_stats_gap.py / test_uncertainties.py
    # but with a dependency-free trapezoid transit.
    rng = numpy.random.default_rng(0)
    start, days = 48, 365.25 * 3
    samples = int(days * 12)
    t = numpy.linspace(start, start + days, samples)
    y = trapezoid(t, 365.25, start + 20, 0.55, 84e-6) + rng.normal(0, 5e-6, samples)
    y[1] = numpy.nan
    dy = None
    if nan_gap:
        y[200:500] = numpy.nan
        t[200:500] = numpy.nan
    if with_dy:
        y[10000:] += rng.normal(0, 5e-5, samples - 10000)
        dy = numpy.full(samples, 5e-6)
        dy[10000:] = 5e-5
    return t, y, dy


def short_lc(n=150):
    rng = numpy.random.default_rng(1)
    t = numpy.linspace(0, 20, n)
    return t, trapezoid(t, 5.0, 0.1, 0.4, 5e-3) + rng.normal(0, 1e-3, n), None


def centered_time():
    rng = numpy.random.default_rng(2)
    t = numpy.arange(-15, 15, 0.01)
    return t, trapezoid(t, 3.3, -1.0, 0.12, 1.5e-3) + rng.normal(0, 4e-4, len(t)), None


def plato_window():
    """PLATO-like high cadence: 27 d at 25 s (93k points), 200 ppm transit."""
    rng = numpy.random.default_rng(3)
    t = numpy.arange(0, 27, 25 / 86400)
    y = trapezoid(t, 3.1, 0.8, 0.11, 2e-4) + rng.normal(0, 3e-4, len(t))
    return t, y, numpy.full(len(t), 3e-4)


def hetero_dy():
    rng = numpy.random.default_rng(4)
    t = numpy.arange(0, 40, 0.02)
    dy = rng.uniform(2e-4, 1.5e-3, len(t))
    y = trapezoid(t, 4.4, 1.7, 0.15, 1.2e-3) + rng.normal(0, 1, len(t)) * dy
    return t, y, dy


Q = dict(show_progress_bar=False, verbose=False)

# name: (loader, power kwargs)
CASES = {
    "syn_earth": (
        lambda: synthetic_earth(),
        dict(
            period_min=360,
            period_max=370,
            oversampling_factor=5,
            duration_grid_step=1.02,
            use_threads=1,
        ),
    ),
    "syn_earth_gap": (
        lambda: synthetic_earth(nan_gap=True),
        dict(
            period_min=360,
            period_max=370,
            oversampling_factor=2,
            duration_grid_step=1.1,
            T0_fit_margin=1.2,
        ),
    ),
    "syn_earth_dy": (
        lambda: synthetic_earth(with_dy=True),
        dict(
            period_min=360,
            period_max=370,
            oversampling_factor=3,
            duration_grid_step=1.05,
            T0_fit_margin=0.2,
        ),
    ),
    "no_fit": (
        lambda: synthetic_earth(),
        dict(
            transit_depth_min=1e-3,
            period_min=360,
            period_max=370,
            oversampling_factor=5,
            duration_grid_step=1.02,
            T0_fit_margin=0.1,
        ),
    ),
    "k2_box": (
        lambda: load_repo("EPIC206154641.csv") + (None,),
        dict(transit_template="box", period_max=6),
    ),
    "k2_grazing": (
        lambda: load_repo("EPIC206154641.csv") + (None,),
        dict(transit_template="grazing", period_max=6),
    ),
    "k2_3_window": (
        lambda: load_repo("EPIC201367065.csv") + (None,),
        dict(period_min=5, period_max=15),
    ),
    "k2_3_threads1": (
        lambda: load_repo("EPIC201367065.csv") + (None,),
        dict(period_min=9, period_max=11, use_threads=1),
    ),
    "short_lc": (short_lc, dict()),
    "centered_time": (centered_time, dict(period_min=2, period_max=5, T0_fit_margin=0)),
    "custom_nonlinear": (
        centered_time,
        dict(
            period_min=2,
            period_max=5,
            rp=0.05,
            a=15,
            per=5,
            limb_dark="nonlinear",
            u=[0.5, 0.1, 0.1, -0.1],
        ),
    ),
    "custom_ecc_squareroot": (
        centered_time,
        dict(
            period_min=2,
            period_max=5,
            ecc=0.3,
            w=40,
            b=0.4,
            limb_dark="squareroot",
            u=[0.3, 0.3],
        ),
    ),
    "narrow_stellar": (
        hetero_dy,
        dict(
            R_star=1,
            R_star_min=0.8,
            R_star_max=1.2,
            M_star=1,
            M_star_min=0.8,
            M_star_max=1.2,
            oversampling_factor=2,
            duration_grid_step=1.05,
            T0_fit_margin=0,
        ),
    ),
    "plato_25s_window": (plato_window, dict(period_min=3.0, period_max=3.2)),
}


def run(name):
    from transitleastsquares import tls_constants, transitleastsquares

    if os.environ.get("TLS_TRANSIT_MODEL"):  # e.g. "batman" to check golden v1
        tls_constants.TRANSIT_MODEL = os.environ["TLS_TRANSIT_MODEL"]

    loader, kw = CASES[name]
    t, y, dy = loader()
    kwargs = dict(Q)
    # The golden set captures the exact statistic: use an exact backend
    kwargs["backend"] = os.environ.get("TLS_GOLDEN_BACKEND", "fused")
    kwargs.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transitleastsquares(t, y, dy, verbose=False).power(**kwargs)


def to_arrays(results):
    out = {}
    for k, v in results.items():
        if v is None:
            v = numpy.nan
        out[k] = numpy.asarray(v, dtype=float)
    return out


def compare(name, results, rtol=1e-9, atol=1e-12):
    """Return list of (key, message) mismatches vs. the golden file."""
    ref = numpy.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    cur = to_arrays(results)
    bad = []
    for k in ref.files:
        if k not in cur:
            bad.append((k, "missing"))
            continue
        a, b = cur[k], ref[k]
        if a.shape != b.shape:
            bad.append((k, f"shape {a.shape} != {b.shape}"))
            continue
        # Spectra amplify chi2 rounding (~1e-13 relative) by 1/std(SR) ~ 1e4
        key_atol = 1e-8 if k in ("power", "power_raw", "SR") else atol
        if not numpy.allclose(a, b, rtol=rtol, atol=key_atol, equal_nan=True):
            d = numpy.nanmax(numpy.abs(a - b)) if a.size else 0
            bad.append((k, f"max abs diff {d:.3e}"))
    for k in cur:
        if k not in ref.files:
            bad.append((k, "new key"))
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        import transitleastsquares

        print("writing golden set", GOLDEN_SET, "from", transitleastsquares.__file__)
        for name in CASES:
            r = run(name)
            numpy.savez_compressed(
                os.path.join(GOLDEN_DIR, name + ".npz"), **to_arrays(r)
            )
            print(f"  {name}: P={r.period} SDE={r.SDE}")
    else:
        for name in CASES:
            bad = compare(name, run(name))
            print(name, "OK" if not bad else bad)
