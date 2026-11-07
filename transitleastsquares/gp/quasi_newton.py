"""Bounded quasi-Newton minimisation (projected BFGS), without a SciPy
dependency, for the gradient fits of the baseline hyperparameters (few
parameters: a dense inverse Hessian).

Each iteration takes the BFGS direction in the free variables (those not
held at a bound by the gradient), steps along the projection of that
direction onto the box with a backtracking Armijo line search (safeguarded
quadratic interpolation) and updates the inverse Hessian in the coordinates
off the bounds if the curvature is positive. It stops on a projected gradient below gtol, or a relative
reduction of the function below ftol (the criteria of SciPy's L-BFGS-B,
whose defaults it shares).
"""

import numpy as np

from transitleastsquares.gp.nelder_mead import Result


def minimize_bounded(fg, x0, lower, upper, maxiter=1000, gtol=1e-3, ftol=2.2e-9, max_ls=30):
    """Minimise f within lower <= x <= upper. fg(x) -> (f, gradient); a
    non-finite f counts as worse than any finite one (the line search steps
    back). Returns a Result (x, fun, nfev = calls of fg, nit)."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(x0, float), lo, hi)
    n = len(x)
    f, g = fg(x)
    nfev, nit = 1, 0
    if not np.isfinite(f):
        return Result(x, f, nfev, nit)
    H = np.eye(n)
    scaled = False
    for nit in range(1, maxiter + 1):
        free = ~(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0)))
        if not free.any() or np.max(np.abs(g[free])) <= gtol:
            break
        d = np.zeros(n)
        d[free] = -H[np.ix_(free, free)] @ g[free]
        if g @ d >= 0:  # not a descent direction: restart from the gradient
            H = np.eye(n)
            scaled = False
            d[free] = -g[free]
        step = 1.0 if scaled else min(1.0, 1.0 / np.linalg.norm(d))  # first step: unit length
        for _ in range(max_ls):
            xn = np.clip(x + step * d, lo, hi)
            fn, gn = fg(xn)
            nfev += 1
            slope = g @ (xn - x)
            if np.isfinite(fn) and fn <= f + 1e-4 * slope:
                break
            if np.isfinite(fn) and fn - f - slope > 0:  # minimum of the quadratic through f, slope and fn
                step = min(max(-0.5 * slope * step / (fn - f - slope), 0.1 * step), 0.5 * step)
            else:
                step *= 0.1
        else:
            break  # no decrease along the direction
        s, yv = xn - x, gn - g
        # coordinates held at a bound, or clipped onto one, take no part in the update: their gradient change
        # (with no step) spoiled the curvature, so that a fit with a parameter on its bound crawled (903 instead of
        # 82 gradients on a Talens light curve, bench/gradfit)
        held = ~free | (xn <= lo) | (xn >= hi)
        s, yv = np.where(held, 0.0, s), np.where(held, 0.0, yv)
        f_old = f
        x, f, g = xn, fn, gn
        if f_old - f <= ftol * max(abs(f_old), abs(f), 1.0):
            break
        sy = s @ yv
        if np.any(s) and sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(yv):
            if not scaled:
                H = (sy / (yv @ yv)) * np.eye(n)
                scaled = True
            rho = 1.0 / sy
            V = np.eye(n) - rho * np.outer(s, yv)
            H = V @ H @ V.T + rho * np.outer(s, s)
    return Result(x, f, nfev, nit)
