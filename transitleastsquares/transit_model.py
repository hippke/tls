"""Limb-darkened transit light curves (replacement for the batman dependency).

Self-contained numpy implementation written for TLS (MIT license):

* Orbit: sky-projected star-planet separation z(t) for circular and eccentric
  orbits (Kepler's equation, Newton iteration). Parameters follow the batman
  conventions: t0 = time of inferior conjunction (mid-transit), inc and w in
  degrees, a and rp in units of the stellar radius.
* Occultation: the blocked flux is integrated exactly in angle and numerically
  in radius. For a stellar annulus of radius r, the arc inside the planet disk
  (radius p, centre distance z) subtends 2*alpha(r) with
      alpha = pi                                   for r <= p - z
      alpha = arccos((r^2 + z^2 - p^2) / (2 r z))  for |z - p| < r < z + p,
  so  blocked(z) = int I(r) 2 alpha(r) r dr.  The radial integral is split at
  the kinks of alpha(r), and each piece uses Gauss-Legendre quadrature after
  the substitution r = lo + (hi - lo)(1 - cos theta)/2, which removes the
  square-root endpoint singularities (at |z - p|, z + p and at the stellar
  limb, where mu = sqrt(1 - r^2)). This converges to ~1e-12 relative accuracy
  with the default number of nodes and works for every limb-darkening law.

Supported laws (I(mu)/I(1), mu = sqrt(1 - r^2)), as in batman:
    uniform      1
    linear       1 - c1 (1 - mu)
    quadratic    1 - c1 (1 - mu) - c2 (1 - mu)^2
    squareroot   1 - c1 (1 - mu) - c2 (1 - sqrt(mu))
    logarithmic  1 - c1 (1 - mu) - c2 mu ln(mu)
    exponential  1 - c1 (1 - mu) - c2 / (1 - exp(mu))
    power2       1 - c1 (1 - mu^c2)
    nonlinear    1 - c1 (1 - mu^1/2) - c2 (1 - mu) - c3 (1 - mu^3/2) - c4 (1 - mu^2)
"""

import functools

import numpy

N_COEFFICIENTS = {
    "uniform": 0,
    "linear": 1,
    "quadratic": 2,
    "squareroot": 2,
    "logarithmic": 2,
    "exponential": 2,
    "power2": 2,
    "nonlinear": 4,
}

# Gauss-Legendre nodes per radial sub-interval (occultation) and for the
# normalisation (total stellar flux). Validated in tests/test_transit_model.py.
NODES_OCCULTATION = 96
NODES_NORMALISATION = 512


def _check_law(u, limb_dark):
    if limb_dark not in N_COEFFICIENTS:
        raise ValueError(
            f"Unknown limb darkening law {limb_dark!r}. "
            f"Supported: {', '.join(N_COEFFICIENTS)}"
        )
    u = [] if u is None else [float(x) for x in numpy.atleast_1d(u)]
    n = N_COEFFICIENTS[limb_dark]
    if limb_dark == "uniform":
        return ()
    if len(u) != n:
        raise ValueError(
            f"Limb darkening law {limb_dark!r} needs {n} coefficient(s), got {len(u)}"
        )
    return tuple(u)


def intensity(mu, u, limb_dark):
    """Normalised specific intensity I(mu)/I(mu=1)"""
    u = _check_law(u, limb_dark)
    mu = numpy.asarray(mu, dtype=float)
    if limb_dark == "uniform":
        return numpy.ones_like(mu)
    if limb_dark == "linear":
        return 1 - u[0] * (1 - mu)
    if limb_dark == "quadratic":
        return 1 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2
    if limb_dark == "squareroot":
        return 1 - u[0] * (1 - mu) - u[1] * (1 - numpy.sqrt(mu))
    if limb_dark == "logarithmic":
        with numpy.errstate(divide="ignore", invalid="ignore"):
            log_term = numpy.where(mu > 0, mu * numpy.log(mu), 0.0)
        return 1 - u[0] * (1 - mu) - u[1] * log_term
    if limb_dark == "exponential":
        with numpy.errstate(divide="ignore"):
            return 1 - u[0] * (1 - mu) - u[1] / (1 - numpy.exp(mu))
    if limb_dark == "power2":
        return 1 - u[0] * (1 - mu ** u[1])
    if limb_dark == "nonlinear":
        return (
            1
            - u[0] * (1 - mu**0.5)
            - u[1] * (1 - mu)
            - u[2] * (1 - mu**1.5)
            - u[3] * (1 - mu**2)
        )
    raise AssertionError  # unreachable


@functools.lru_cache(maxsize=8)
def _cosine_nodes(n):
    """Nodes s in (0, 1) and weights w for int_0^1 f(s) ds with the substitution
    s = (1 - cos theta)/2 (theta in (0, pi)), via Gauss-Legendre in theta."""
    x, wx = numpy.polynomial.legendre.leggauss(n)
    theta = 0.5 * numpy.pi * (x + 1)
    s = 0.5 * (1 - numpy.cos(theta))
    w = wx * 0.5 * numpy.pi * 0.5 * numpy.sin(theta)  # dtheta/dx * ds/dtheta
    return s, w


def _radial_intensity(r, u, limb_dark):
    mu = numpy.sqrt(numpy.clip(1 - r * r, 0.0, None))
    return intensity(mu, u, limb_dark)


@functools.lru_cache(maxsize=64)
def total_flux(u, limb_dark):
    """Disk-integrated flux 2 pi int_0^1 I(r) r dr (u: tuple of coefficients)"""
    s, w = _cosine_nodes(NODES_NORMALISATION)
    r = s  # interval [0, 1]
    return 2 * numpy.pi * numpy.sum(w * _radial_intensity(r, u, limb_dark) * r)


def occulted_fraction(z, p, u, limb_dark, nodes=NODES_OCCULTATION):
    """Fraction of the stellar flux blocked by a disk of radius p (stellar radii)
    at projected separations z (array). 0 outside of transit."""
    u = _check_law(u, limb_dark)
    z = numpy.abs(numpy.asarray(z, dtype=float))
    p = float(p)
    result = numpy.zeros_like(z)
    inside = z < 1 + p
    if p <= 0 or not numpy.any(inside):
        return result
    zi = z[inside]
    s, w = _cosine_nodes(nodes)
    blocked = numpy.zeros_like(zi)

    # (A) annuli completely covered by the planet: r in [0, min(p - z, 1)]
    hi_a = numpy.clip(p - zi, 0.0, 1.0)
    has_a = hi_a > 0
    if numpy.any(has_a):
        h = hi_a[has_a, None]
        r = h * s[None, :]
        integrand = _radial_intensity(r, u, limb_dark) * r
        blocked[has_a] += 2 * numpy.pi * numpy.sum(w * integrand, axis=1) * h[:, 0]

    # (B) partially covered annuli: r in [|z - p|, min(z + p, 1)]
    lo_b = numpy.abs(zi - p)
    hi_b = numpy.minimum(zi + p, 1.0)
    has_b = (hi_b > lo_b) & (zi > 0)
    if numpy.any(has_b):
        lo = lo_b[has_b, None]
        width = hi_b[has_b, None] - lo
        zb = zi[has_b, None]
        r = lo + width * s[None, :]
        cos_alpha = numpy.clip((r * r + zb * zb - p * p) / (2 * r * zb), -1.0, 1.0)
        integrand = _radial_intensity(r, u, limb_dark) * 2 * numpy.arccos(cos_alpha) * r
        blocked[has_b] += numpy.sum(w * integrand, axis=1) * width[:, 0]

    result[inside] = blocked / total_flux(u, limb_dark)
    return result


def _solve_kepler(M, ecc, tol=1e-12, max_iter=50):
    """Eccentric anomaly E from mean anomaly M (Newton iteration)"""
    E = M + ecc * numpy.sin(M) if ecc < 0.8 else numpy.full_like(M, numpy.pi)
    for _ in range(max_iter):
        dE = (E - ecc * numpy.sin(E) - M) / (1 - ecc * numpy.cos(E))
        E = E - dE
        if numpy.max(numpy.abs(dE)) < tol:
            break
    return E


def sky_separation(t, t0, per, a, inc, ecc=0.0, w=90.0):
    """Projected star-planet separation (stellar radii). Points where the planet
    is behind the star (secondary eclipse side) are set to +inf."""
    t = numpy.asarray(t, dtype=float)
    inc_rad = numpy.radians(inc)
    w_rad = numpy.radians(w)
    if ecc == 0:
        phase = 2 * numpy.pi * (t - t0) / per  # f + w - pi/2
        x = a * numpy.sin(phase)
        y = a * numpy.cos(phase) * numpy.cos(inc_rad)
        z = numpy.sqrt(x * x + y * y)
        front = numpy.cos(phase) > 0
    else:
        # Time of periastron from the true anomaly at inferior conjunction
        f_conj = numpy.pi / 2 - w_rad
        E_conj = 2 * numpy.arctan(
            numpy.sqrt((1 - ecc) / (1 + ecc)) * numpy.tan(f_conj / 2)
        )
        M_conj = E_conj - ecc * numpy.sin(E_conj)
        tp = t0 - per * M_conj / (2 * numpy.pi)
        M = numpy.mod(2 * numpy.pi * (t - tp) / per, 2 * numpy.pi)
        E = _solve_kepler(M, ecc)
        f = 2 * numpy.arctan2(
            numpy.sqrt(1 + ecc) * numpy.sin(E / 2),
            numpy.sqrt(1 - ecc) * numpy.cos(E / 2),
        )
        r = a * (1 - ecc * numpy.cos(E))
        z = r * numpy.sqrt(1 - numpy.sin(w_rad + f) ** 2 * numpy.sin(inc_rad) ** 2)
        front = numpy.sin(w_rad + f) > 0
    return numpy.where(front, z, numpy.inf)


def light_curve(t, t0, per, rp, a, inc, ecc, w, u, limb_dark):
    """Relative flux of a transit (1 = out of transit) at times t."""
    _check_law(u, limb_dark)
    z = sky_separation(t, t0, per, a, inc, ecc, w)
    return 1 - occulted_fraction(z, rp, u, limb_dark)
