"""Baseline models: Gaussian processes for the stellar (and instrumental)
variability that is marginalised in the transit search.

Public units: time in days, flux normalised to ~1, amplitudes (sigma,
sigma_slow, jitter) in relative flux. Internally the state-space code works
in ppt.
"""

import math

import numba
import numpy as np

from transitleastsquares.gp import fitting
from transitleastsquares.gp.statespace import gp_mean, log_likelihood, rotation_model, spline_model

PPT = 1e3


def robust_errors(y):
    """Per-point flux uncertainty from the point-to-point scatter (robust
    against slow variability and outliers), for data without dy."""
    d = np.diff(y)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2)


@numba.njit(cache=True, parallel=True)
def _ls_power(t, yc, freqs):
    """Lomb-Scargle power at each frequency (in parallel over frequencies):
    one pass over the data with the sums C = y cos, S = y sin, CC, SS, CS
    of wt; the phase w tau = atan2(2 CS, CC - SS) / 2 rotates them."""
    power = np.empty(len(freqs))
    for i in numba.prange(len(freqs)):
        w = 2 * math.pi * freqs[i]
        C = S = CC = SS = CS = 0.0
        for k in range(len(t)):
            c, s = math.cos(w * t[k]), math.sin(w * t[k])
            C += yc[k] * c
            S += yc[k] * s
            CC += c * c
            SS += s * s
            CS += c * s
        phi = 0.5 * math.atan2(2 * CS, CC - SS)
        cp, sp = math.cos(phi), math.sin(phi)
        yc_c, yc_s = C * cp + S * sp, S * cp - C * sp
        cc = CC * cp * cp + 2 * CS * cp * sp + SS * sp * sp
        ss = SS * cp * cp - 2 * CS * cp * sp + CC * sp * sp
        power[i] = yc_c * yc_c / cc + yc_s * yc_s / ss
    return power


def rotation_period_guess(t, y, period_min=0.2, period_max=20.0, oversample=5):
    """Peak of the Lomb-Scargle periodogram, as the starting point of the
    rotation fit. The cost is (points x frequencies), both growing with the
    span: data with a cadence below period_min / 16 are binned to
    period_min / 8 first (8 samples per shortest period)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    width = period_min / 8
    if len(t) > 1 and np.median(np.diff(t)) < 0.5 * width:
        t, y, _, _ = fitting.bin_series(t, y, np.ones(len(y)), width)
    yc = y - np.mean(y)
    span = t.max() - t.min()
    df = 1 / (oversample * span)
    freqs = np.arange(1 / period_max, 1 / period_min, df)
    return 1 / freqs[np.argmax(_ls_power(t, yc, freqs))]


FIT_BIN_WIDTH = 10.0 / 1440  # days; rotation fits bin data with a cadence below half of it


class Baseline:
    """A Gaussian-process model of the baseline (base class)."""

    def _model(self, t, dy):
        raise NotImplementedError

    def log_likelihood(self, t, y, dy):
        """Gaussian log likelihood of the flux y (relative units) under this
        baseline plus white noise dy, with the GP integrated out."""
        t, y, dy = (np.asarray(a, float) for a in (t, y, dy))
        ll_ppt = log_likelihood(self._model(t, dy), t, (y - 1) * PPT, dy * PPT)
        return ll_ppt + (len(t) - self._model(t, dy).n_diffuse) * math.log(PPT)

    def trend(self, t, y, dy, use=None, threads=None, device="cpu"):
        """Posterior mean of the baseline at all t (relative flux),
        conditioned on the points with use=True (default: all); in up to
        `threads` threads (default: numba's thread count). device="gpu": as
        f = y - R C^-1 y without the smoother (transitleastsquares.gpu)."""
        t, y, dy = (np.asarray(a, float) for a in (t, y, dy))
        use = np.ones(len(t), bool) if use is None else np.asarray(use, bool)
        yerr = np.where(use, dy, 1e6) * PPT  # excluded points carry no information
        threads = numba.get_num_threads() if threads is None else int(threads)
        if device == "gpu":
            from transitleastsquares.gp.statespace import n_segments
            from transitleastsquares.gpu import driver

            model = self._model(t, dy)
            yp = (y - 1) * PPT * use
            return 1 + driver.trend_gpu(model, t, yp, yerr**2 + model.jitter**2, n_segments(len(t), threads)) / PPT
        return 1 + gp_mean(self._model(t, dy), t, (y - 1) * PPT * use, yerr, threads) / PPT

    def as_dict(self):
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items())
        return f"{type(self).__name__}({args})"

class RotationBaseline(Baseline):
    """Rotation GP (celerite2's RotationTerm: damped oscillators at the
    rotation period and half of it), a slow non-periodic damped oscillator
    (Q = 1/sqrt(2)), a free mean, and white-noise jitter added to dy.

    sigma, sigma_slow, jitter: relative flux; period, rho_slow: days;
    Q0, dQ, f: dimensionless."""

    KEYS = ("sigma", "period", "Q0", "dQ", "f", "sigma_slow", "rho_slow", "jitter")
    _AMPLITUDES = ("sigma", "sigma_slow", "jitter")

    def __init__(self, sigma, period, Q0, dQ, f, sigma_slow, rho_slow, jitter=0.0):
        values = dict(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f,
                      sigma_slow=sigma_slow, rho_slow=rho_slow, jitter=jitter)
        # stored exactly as used internally (ppt), so that fits and warm
        # starts see bit-identical values
        self._p = {k: (v * PPT if k in self._AMPLITUDES else v) for k, v in values.items()}

    def __getattr__(self, name):
        p = self.__dict__.get("_p")
        if p is None or name not in p:
            raise AttributeError(name)
        return p[name] / PPT if name in self._AMPLITUDES else p[name]

    def as_dict(self):
        return {k: getattr(self, k) for k in self.KEYS}

    def __eq__(self, other):
        return type(self) is type(other) and self._p == other._p

    def _model(self, t, dy):
        return rotation_model(self._p)

    @classmethod
    def _from_ppt(cls, p):
        b = cls.__new__(cls)
        b._p = {k: p[k] for k in cls.KEYS}
        return b

    @classmethod
    def fit(cls, t, y, dy, period_guess=None, use=None, n_starts=1, use_threads=None, start=None,
            bin_width=FIT_BIN_WIDTH, passes=1, tol=1e-3):
        """Type-II maximum likelihood. period_guess (default: Lomb-Scargle
        peak); use: points to fit (default all); n_starts (1-4, default 1) starting
        points in up to use_threads threads (default: numba's thread count);
        start: a previous RotationBaseline (single pass from there);
        bin_width: fit on data binned to this width (days, default 10 min)
        if the cadence is shorter than half of it (None: never bin);
        passes, tol: Nelder-Mead passes from a cold start and their
        tolerance (default 1 and 1e-3).
        The hyperparameters are nuisance parameters of the search: the
        likelihood is flat and multimodal, and the defaults find the same
        planets as passes=3, tol=1e-4 (400 K2 injections) at 2x less fit
        time, but not always the same optimum. For better optima (e.g. to
        use the rotation period), fit with n_starts=4, passes=3, tol=1e-4
        and pass the baseline to gp.search.
        Returns (baseline, log likelihood of the unbinned data)."""
        t, y, dy = (np.asarray(a, float) for a in (t, y, dy))
        if period_guess is None:
            m = slice(None) if use is None else np.asarray(use, bool)
            period_guess = rotation_period_guess(t[m], y[m])
        p, ll_ppt = fitting.fit_rotation(
            t, (y - 1) * PPT, dy * PPT, period_guess, mask=use,
            start=None if start is None else start._p, n_starts=n_starts, use_threads=use_threads,
            bin_width=bin_width, passes=passes, tol=tol,
        )
        n = len(t) if use is None else int(np.sum(use))
        return cls._from_ppt(p), ll_ppt + (n - 1) * math.log(PPT)


class SplineBaseline(Baseline):
    """Cubic smoothing spline as a GP (integrated Wiener process), with
    smoothing length h (days; the equivalent-kernel width), a free mean and
    white-noise jitter (relative flux). sigma_ref (relative flux) and cadence
    (days) fix the process noise q = sigma_ref^2 cadence / h^4; by default
    they are taken from the data (median dy, median time step) when used."""

    def __init__(self, h, jitter=0.0, sigma_ref=None, cadence=None):
        self.h, self.cadence = h, cadence
        self._jitter = jitter * PPT  # internal units (ppt), see RotationBaseline
        self._sigma_ref = None if sigma_ref is None else sigma_ref * PPT

    jitter = property(lambda self: self._jitter / PPT)
    sigma_ref = property(lambda self: None if self._sigma_ref is None else self._sigma_ref / PPT)

    def as_dict(self):
        return dict(h=self.h, jitter=self.jitter, sigma_ref=self.sigma_ref, cadence=self.cadence)

    def __eq__(self, other):
        return type(self) is type(other) and (self.h, self._jitter, self._sigma_ref, self.cadence) == \
            (other.h, other._jitter, other._sigma_ref, other.cadence)

    def _model(self, t, dy):
        sig = np.median(dy) * PPT if self._sigma_ref is None else self._sigma_ref
        cad = np.median(np.diff(t)) if self.cadence is None else self.cadence
        return spline_model(self.h, sig, cad, self._jitter)

    @classmethod
    def fit(cls, t, y, dy, use=None, h_range=(0.05, 5.0), use_threads=None, start=None):
        """Type-II maximum likelihood for h and the jitter (3 starts in up to
        use_threads threads, or a single pass from start, a previous
        SplineBaseline). Returns (baseline, log likelihood)."""
        t, y, dy = (np.asarray(a, float) for a in (t, y, dy))
        s = None if start is None else dict(h=start.h, jitter=start._jitter)
        p, ll_ppt = fitting.fit_spline(t, (y - 1) * PPT, dy * PPT, mask=use, h_range=h_range,
                                       start=s, use_threads=use_threads)
        b = cls(p["h"], cadence=p["cadence"])
        b._jitter, b._sigma_ref = p["jitter"], p["sigma"]
        n = len(t) if use is None else int(np.sum(use))
        return b, ll_ppt + (n - 3) * math.log(PPT)


def fit_with_flare_clipping(kind, t, y, dy, nsigma=4.0, max_iter=5, **fit_kwargs):
    """Fit a baseline ("rotation" or "spline") and iteratively remove
    positive outliers (flares) of its residuals against the trend of the
    points kept so far; dips (transits) are never
    clipped. Each round is warm-started from the previous one. A round that
    would remove more than 20 % of the points, or a degenerate trend, ends
    the clipping. Returns (baseline, log likelihood, use) with use = points
    kept."""
    cls = {"rotation": RotationBaseline, "spline": SplineBaseline}[kind]
    if kind == "rotation" and fit_kwargs.get("period_guess") is None:
        # once, before clipping: it also sets the fit's period bounds
        fit_kwargs["period_guess"] = rotation_period_guess(t, y)
    use = np.ones(len(t), bool)
    b = None
    for _ in range(max_iter):
        b, ll = cls.fit(t, y, dy, use=use, start=b, **fit_kwargs)
        resid = y - b.trend(t, y, dy, use)  # conditioned on the kept points: a clipped flare no longer pulls the trend
        r = resid[use]
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not (np.isfinite(mad) and mad > 0):
            break  # degenerate trend: keep the current points
        new = use & (resid < nsigma * mad)
        if new.sum() == use.sum() or new.sum() < 0.8 * use.sum():
            break  # converged, or a round would remove > 20 % (not flares)
        use = new
    return b, ll, use
