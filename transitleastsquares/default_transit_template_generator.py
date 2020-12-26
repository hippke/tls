from abc import ABC, abstractmethod

import batman
import numpy

from transitleastsquares import tls_constants
from transitleastsquares.grid import T14
from transitleastsquares.interpolation import interp1d
from transitleastsquares.transit_template_generator import TransitTemplateGenerator


class DefaultTransitTemplateGenerator(TransitTemplateGenerator):
    def __init__(self):
        super().__init__()

    def reference_transit(self, period_grid, duration_grid, samples, per, rp, a, inc, ecc, w, u, limb_dark):
        f = numpy.ones(tls_constants.SUPERSAMPLE_SIZE)
        duration = 1  # transit duration in days. Increase for exotic cases
        t = numpy.linspace(-duration * 0.5, duration * 0.5, tls_constants.SUPERSAMPLE_SIZE)
        ma = batman.TransitParams()
        ma.t0 = 0  # time of inferior conjunction
        ma.per = per  # orbital period, use Earth as a reference
        ma.rp = rp  # planet radius (in units of stellar radii)
        ma.a = a  # semi-major axis (in units of stellar radii)
        ma.inc = inc  # orbital inclination (in degrees)
        ma.ecc = ecc  # eccentricity
        ma.w = w  # longitude of periastron (in degrees)
        ma.u = u  # limb darkening coefficients
        ma.limb_dark = limb_dark  # limb darkening model
        m = batman.TransitModel(ma, t)  # initializes model
        flux = m.light_curve(ma)  # calculates light curve
        # Determine start of transit (first value < 1)
        idx_first = numpy.argmax(flux < 1)
        intransit_time = t[idx_first: -idx_first + 1]
        intransit_flux = flux[idx_first: -idx_first + 1]
        # Downsample (bin) to target sample size
        x_new = numpy.linspace(t[idx_first], t[-idx_first - 1], samples, per)
        f = interp1d(x_new, intransit_time)
        downsampled_intransit_flux = f(intransit_flux)
        # Rescale to height [0..1]
        rescaled = (numpy.min(downsampled_intransit_flux) - downsampled_intransit_flux) / (
                numpy.min(downsampled_intransit_flux) - 1
        )
        return rescaled

    def duration_grid(self, periods, shortest, log_step=tls_constants.DURATION_GRID_STEP):
        duration_max = T14(
            R_s=tls_constants.R_STAR_MAX,
            M_s=tls_constants.M_STAR_MAX,
            P=min(periods),
            small=False  # large planet for long transit duration
        )
        duration_min = T14(
            R_s=tls_constants.R_STAR_MIN,
            M_s=tls_constants.M_STAR_MIN,
            P=max(periods),
            small=True  # small planet for short transit duration
        )

        durations = [duration_min]
        current_depth = duration_min
        while current_depth * log_step < duration_max:
            current_depth = current_depth * log_step
            durations.append(current_depth)
        durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
        return durations

    def min_duration(self, period, R_star, M_star, periods=None):
        return T14(R_s=R_star, M_s=M_star, P=period, small=True)

    def max_duration(self, period, R_star, M_star, periods=None):
        return T14(R_s=R_star, M_s=M_star, P=period, small=True)
