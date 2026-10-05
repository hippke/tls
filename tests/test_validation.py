import numpy
import pytest

from transitleastsquares import transitleastsquares


@pytest.mark.parametrize("threads", [0, "1", -1, 1.5])
def test_invalid_use_threads(threads):
    t = numpy.linspace(0, 1, 2000)
    y = numpy.ones_like(t)
    with pytest.raises(ValueError):
        transitleastsquares(t, y, verbose=False).power(
            use_threads=threads, show_progress_bar=False
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(R_star=-1),
        dict(R_star_min=2.0),
        dict(M_star_max=0.5),
        dict(period_min=-1),
        dict(period_min=5, period_max=4),
        dict(n_transits_min=1.5),
        dict(n_transits_min=0),
        dict(transit_template="nonexistent"),
        dict(backend="nonexistent"),
        dict(SDE_detrend="nonexistent"),
    ],
)
def test_invalid_parameters(kwargs):
    t = numpy.linspace(0, 20, 2000)
    y = numpy.ones_like(t)
    with pytest.raises(ValueError):
        transitleastsquares(t, y, verbose=False).power(
            show_progress_bar=False, **kwargs
        )
