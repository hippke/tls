"""Tests of TLS' own transit model (transit_model.py), which replaced batman.

References:
* batman (if installed): the model used by TLS <= 1.33,
* adaptive quadrature (scipy.integrate.quad) of the occultation integral,
* exact analytic results (uniform disk: circle-circle intersection area;
  disk-integrated flux of polynomial laws).
"""

import numpy
import pytest

from transitleastsquares import tls_constants
from transitleastsquares import transit_model as tm
from transitleastsquares.transit import reference_transit

LAWS = {
    "uniform": [],
    "linear": [0.5],
    "quadratic": [0.4804, 0.1867],
    "squareroot": [0.3, 0.3],
    "logarithmic": [0.5, 0.2],
    "exponential": [0.3, 0.1],
    "power2": [0.6, 0.6],
    "nonlinear": [0.5, 0.1, 0.1, -0.1],
}

# (rp, a, inc, ecc, w): TLS default template, box, grazing, partial grazing
# (b > 1 - p), large planet, eccentric orbits with several arguments of periastron
GEOMETRIES = [
    (0.03, 23.1, 89.21, 0.0, 90.0),
    (0.1, 26.9, 90.0, 0.0, 90.0),
    (0.03, 23.1, numpy.degrees(numpy.arccos(0.99 / 23.1)), 0.0, 90.0),
    (0.1, 20.0, numpy.degrees(numpy.arccos(1.02 / 20.0)), 0.0, 90.0),
    (0.2, 10.0, 87.0, 0.0, 90.0),
    (0.05, 15.0, 88.5, 0.3, 40.0),
    (0.05, 15.0, 89.0, 0.6, 200.0),
    (0.008, 215.0, 89.95, 0.1, 300.0),
]
T = numpy.linspace(-0.5, 0.5, 4001)


def batman_flux(t, rp, a, inc, ecc, w, u, law, per=12.9, max_err=1.0):
    batman = pytest.importorskip("batman")
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, per, rp, a, inc, ecc, w
    p.u, p.limb_dark = list(u), law
    return batman.TransitModel(p, t, max_err=max_err).light_curve(p)


# Max |flux difference| allowed against batman. batman evaluates quadratic and
# linear limb darkening analytically with polynomial approximations of the
# elliptic integrals (~1e-8), and the other laws by numerical integration
# (max_err parameter). batman's exponential law is inaccurate near the limb
# (~5e-5, see test_exponential_against_quadrature) and is not compared.
TOLERANCE_VS_BATMAN = {
    "uniform": 1e-12,
    "linear": 3e-8,
    "quadratic": 3e-8,
    "squareroot": 3e-7,
    "logarithmic": 3e-7,
    "power2": 3e-7,
    "nonlinear": 3e-7,
}


@pytest.mark.parametrize("law", list(TOLERANCE_VS_BATMAN))
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_against_batman(law, geometry):
    rp, a, inc, ecc, w = geometry
    u = LAWS[law]
    ours = tm.light_curve(T, 0.0, 12.9, rp, a, inc, ecc, w, u, law)
    ref = batman_flux(T, rp, a, inc, ecc, w, u, law, max_err=0.01)
    # eccentric orbits: batman's Kepler solver is less precise (~1e-10 in flux)
    tol = TOLERANCE_VS_BATMAN[law] + (1e-9 if ecc > 0 else 0.0)
    assert numpy.max(numpy.abs(ours - ref)) < tol
    # identical first/last in-transit sample (relevant for the TLS template)
    assert numpy.argmax(ours < 1) == numpy.argmax(ref < 1)
    assert numpy.argmax(ours[::-1] < 1) == numpy.argmax(ref[::-1] < 1)


def quad_fraction(z, p, u, law):
    from scipy import integrate

    def intensity(r):
        return float(tm.intensity(numpy.sqrt(max(1.0 - r * r, 0.0)), u, law))

    def integrand(r):
        if r <= p - z:
            alpha = numpy.pi
        elif abs(z - p) < r < z + p:
            alpha = numpy.arccos(
                numpy.clip((r * r + z * z - p * p) / (2 * r * z), -1, 1)
            )
        else:
            alpha = 0.0
        return intensity(r) * 2 * alpha * r

    kw = dict(limit=1000, epsabs=1e-15, epsrel=1e-13)
    total = 2 * numpy.pi * integrate.quad(lambda r: intensity(r) * r, 0, 1, **kw)[0]
    points = sorted({x for x in (abs(z - p), z + p, p - z) if 0 < x < 1})
    blocked = integrate.quad(integrand, 0, 1, points=points or None, **kw)[0]
    return blocked / total


@pytest.mark.parametrize("law", list(LAWS))
@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_against_adaptive_quadrature(law, p):
    pytest.importorskip("scipy")
    u = LAWS[law]
    zs = numpy.array([0.0, 0.5 * p, p, 0.3, 0.7, 1 - p, 1 - 0.5 * p, 1.0, 1 + 0.5 * p])
    ours = tm.occulted_fraction(zs, p, u, law)
    ref = numpy.array([quad_fraction(z, p, u, law) for z in zs])
    numpy.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-12)


def test_exponential_against_quadrature():
    """batman's exponential law deviates by ~5e-5 near the limb; ours matches an
    independent adaptive quadrature."""
    pytest.importorskip("scipy")
    u, p = LAWS["exponential"], 0.2
    for z in [0.0, 0.5, 0.9, 1.05]:
        ours = tm.occulted_fraction(numpy.array([z]), p, u, "exponential")[0]
        assert abs(ours - quad_fraction(z, p, u, "exponential")) < 1e-11


def circle_overlap_area(z, p):
    """Exact area of overlap of the unit disk and a disk (radius p, distance z)."""
    if z >= 1 + p:
        return 0.0
    if z <= abs(1 - p):
        return numpy.pi * min(1.0, p) ** 2
    k0 = numpy.arccos((p * p + z * z - 1) / (2 * p * z))
    k1 = numpy.arccos((1 - p * p + z * z) / (2 * z))
    return p * p * k0 + k1 - 0.5 * numpy.sqrt(4 * z * z - (1 + z * z - p * p) ** 2)


@pytest.mark.parametrize("p", [0.005, 0.05, 0.5, 1.0, 1.5])
def test_uniform_exact(p):
    zs = numpy.linspace(0, 1 + p + 0.1, 301)
    ours = tm.occulted_fraction(zs, p, [], "uniform")
    exact = numpy.array([circle_overlap_area(z, p) for z in zs]) / numpy.pi
    numpy.testing.assert_allclose(ours, exact, rtol=1e-11, atol=1e-13)


def test_total_flux_analytic():
    c1, c2, c3, c4 = 0.5, 0.1, 0.1, -0.1
    pi = numpy.pi
    cases = [
        ("uniform", (), pi),
        ("linear", (c1,), pi * (1 - c1 / 3)),
        ("quadratic", (c1, c2), pi * (1 - c1 / 3 - c2 / 6)),
        ("squareroot", (c1, c2), pi * (1 - c1 / 3 - c2 / 5)),
        ("logarithmic", (c1, c2), pi * (1 - c1 / 3 + 2 * c2 / 9)),
        ("power2", (c1, 0.6), pi * (1 - c1 * 0.6 / (0.6 + 2))),
        (
            "nonlinear",
            (c1, c2, c3, c4),
            pi * (1 - c1 / 5 - c2 / 3 - 3 * c3 / 7 - c4 / 2),
        ),
    ]
    for law, u, exact in cases:
        assert abs(tm.total_flux(u, law) / exact - 1) < 1e-11, law


@pytest.mark.parametrize("law", list(LAWS))
def test_self_convergence(law):
    z = tm.sky_separation(T, 0, 12.9, 10.0, 87.0)
    a = tm.occulted_fraction(z, 0.2, LAWS[law], law)
    b = tm.occulted_fraction(z, 0.2, LAWS[law], law, nodes=1024)
    assert numpy.max(numpy.abs(a - b)) < 1e-11


def test_out_of_transit_exactly_one_and_symmetric():
    f = tm.light_curve(
        T, 0.0, 12.9, 0.03, 23.1, 89.21, 0.0, 90.0, [0.48, 0.19], "quadratic"
    )
    z = tm.sky_separation(T, 0.0, 12.9, 23.1, 89.21)
    assert numpy.all(f[z >= 1.03] == 1.0)
    assert numpy.all(f[z < 1.03] < 1.0)
    numpy.testing.assert_allclose(f, f[::-1], atol=1e-15)


def test_full_occultation():
    assert tm.occulted_fraction(
        numpy.array([0.0, 0.2]), 1.5, [0.5], "linear"
    ) == pytest.approx([1.0, 1.0], abs=1e-12)


def test_secondary_eclipse_side_is_not_a_transit():
    t = numpy.linspace(-6.45, 6.45, 2001)  # one full orbit, P = 12.9
    f = tm.light_curve(t, 0.0, 12.9, 0.1, 5.0, 90.0, 0.0, 90.0, [0.5], "linear")
    assert numpy.all(f[numpy.abs(t) > 3] == 1.0)  # around phase 0.5
    assert f.min() < 0.99


def test_kepler_solver():
    M = numpy.linspace(0, 2 * numpy.pi, 1001)
    for ecc in [0.0, 0.1, 0.5, 0.9, 0.99]:
        E = tm._solve_kepler(M, ecc)
        numpy.testing.assert_allclose(E - ecc * numpy.sin(E), M, atol=1e-11)


def test_invalid_laws():
    with pytest.raises(ValueError):
        tm.light_curve(T, 0, 1, 0.1, 10, 90, 0, 90, [0.5], "nonexistent")
    with pytest.raises(ValueError):
        tm.light_curve(T, 0, 1, 0.1, 10, 90, 0, 90, [0.5], "quadratic")
    with pytest.raises(ValueError):
        tm.light_curve(T, 0, 1, 0.1, 10, 90, 0, 90, [0.5, 0.1, 0.1], "nonlinear")


@pytest.mark.parametrize(
    "params",
    [
        dict(
            per=12.9,
            rp=0.03,
            a=23.1,
            inc=89.21,
            ecc=0,
            w=90,
            u=[0.4804, 0.1867],
            limb_dark="quadratic",
        ),  # TLS default
        dict(
            per=12.9,
            rp=0.03,
            a=23.1,
            inc=numpy.degrees(numpy.arccos(0.99 / 23.1)),
            ecc=0,
            w=90,
            u=[0.4804, 0.1867],
            limb_dark="quadratic",
        ),  # grazing
        dict(
            per=29, rp=0.1, a=26.9, inc=90, ecc=0, w=90, u=[0], limb_dark="linear"
        ),  # box
        dict(
            per=5,
            rp=0.05,
            a=15,
            inc=89,
            ecc=0,
            w=90,
            u=[0.5, 0.1, 0.1, -0.1],
            limb_dark="nonlinear",
        ),
    ],
)
def test_tls_template_matches_batman(params):
    """The normalised TLS reference template (what the search uses)."""
    pytest.importorskip("batman")
    ours = reference_transit(samples=2000, **params)
    old = tls_constants.TRANSIT_MODEL
    try:
        tls_constants.TRANSIT_MODEL = "batman"
        ref = reference_transit(samples=2000, **params)
    finally:
        tls_constants.TRANSIT_MODEL = old
    # The template is normalised to depth 1: batman's ~1e-8 absolute error
    # becomes ~1e-5 for a ~1000 ppm transit; ~1e-6 (max_err) for the
    # numerically integrated laws becomes ~1e-3.
    tol = 1e-4 if params["limb_dark"] in ("quadratic", "linear") else 2e-3
    assert numpy.max(numpy.abs(ours - ref)) < tol
