from abc import ABC, abstractmethod

import batman
import numpy

from transitleastsquares import tls_constants
from transitleastsquares.grid import T14
from transitleastsquares.interpolation import interp1d
from transitleastsquares.transit_template_generator import TransitTemplateGenerator


class CometTransitTemplateGenerator(TransitTemplateGenerator):
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
        flux = self._reference_comet_transit(t, flux, per)
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
        max_period = max(periods)
        # Assuming max comet tail 0.25 the orbit length of a 0.5 days period orbit.
        duration_max = 0.25 * ((0.5 / max_period) ** (2 / 3))
        duration_min = T14(
            R_s=tls_constants.R_STAR_MIN,
            M_s=tls_constants.M_STAR_MIN,
            P=max_period,
            small=True  # small planet for short transit duration
        )

        durations = [duration_min]
        current_depth = duration_min
        while current_depth * log_step < duration_max:
            current_depth = current_depth * log_step
            durations.append(current_depth)
        durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
        return durations

    @staticmethod
    def __reference_comet_transit(t, flux, per):
        idx_first = numpy.argmax(flux < 1)
        t0 = 0.5 * per
        t1 = (t[idx_first] + 0.5) * per
        t4 = (t[-idx_first + 1] + 0.5) * per
        ingress_param = 0.2
        egress_param = 0.2
        amplitude = 1 - numpy.min(flux)
        y = numpy.ones(len(t))
        initialized = False
        for i in range(len(t)):
            time = (t[i] + 0.5) * per
            if flux[i] < 1 and time <= t0:
                y[i] = 1 - amplitude * numpy.exp(-((time - t0) ** 2 / (2 * (ingress_param ** 2))))
                initialized = True
            elif time > t0 and flux[i] < 1:
                y[i] = 1 - amplitude * numpy.exp((t0 - time) / egress_param)
        return y