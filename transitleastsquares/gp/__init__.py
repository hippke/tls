"""Transit search with the stellar variability marginalised (experimental).

Instead of detrending first and searching afterwards, the baseline is a
Gaussian process that is integrated out inside the search: for every trial
transit the depth is fitted jointly with the GP, so the transit is not
suppressed by the detrending and its depth is unbiased.

    from transitleastsquares import gp

    results = gp.search(t, y, dy)                      # rotation GP, fitted
    results = gp.search(t, y, dy, baseline="spline")   # smoothing spline
    b, loglik = gp.RotationBaseline.fit(t, y, dy)     # fit only
    results = gp.search(t, y, dy, baseline=b)         # fixed baseline

The module is not imported by `import transitleastsquares`; the TLS search
itself is unchanged.
"""

from transitleastsquares.gp.baselines import Baseline, RotationBaseline, SplineBaseline
from transitleastsquares.gp.api import GPSearchResults, search

__all__ = ["search", "GPSearchResults", "Baseline", "RotationBaseline", "SplineBaseline"]
