import batman
import numpy

from .. import tls_constants
from ..template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator
from ..results import transitleastsquaresresults
from ..grid import T14
from ..interpolation import interp1d


class TailedTransitTemplateGenerator(DefaultTransitTemplateGenerator):
    """
    This class uses the equation (6) from Kennedy et al. (2019) to model exocomet transits.
    """

    def __init__(self):
        super().__init__()

    def reference_transit(self, period_grid, duration_grid, samples, per, rp, a, inc, ecc, w, u, limb_dark):
        """
        Creates a reference transit with the desired shape
        :param period_grid: The grid of periods to be processed
        :param duration_grid: The grid of durations to be processed
        :param samples: The samples count
        :param per: The period for the template
        :param rp: The radius of the comet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param a: The semimajor axis of the comet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param inc: The inclination of the comet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param ecc: The eccentricity of the comet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param w:
        :param u:
        :param limb_dark: The limb darkening applied to the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        """
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
        flux = self.__reference_comet_transit(t, flux, per)
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
        duration_max = self.max_duration(min(periods), tls_constants.R_STAR_MAX, tls_constants.M_STAR_MAX, periods)
        duration_min = self.min_duration(max(periods), tls_constants.R_STAR_MIN, tls_constants.M_STAR_MIN, periods)
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
        """
        Max transit duration of an exocomet cannot be enforced and we choose a soft limit of 10 times the largest
        transit duration given by T14.
        :param period: The period for which the duration needs to be calculated.
        :param R_star: The radius of the host star.
        :param M_star: The mass of the host star
        :param periods: The period grid.
        """
        t14 = T14(R_s=R_star, M_s=M_star, P=period, small=False) * 10
        return t14 if t14 < 1 else 0.99

    def calculate_results(self, no_transits_were_fit, chi2, chi2red, chi2_min, chi2red_min, test_statistic_periods,
                          test_statistic_depths, transitleastsquares, lc_arr, best_row, period_grid, durations,
                          duration, maxwidth_in_samples):
        results = super().calculate_results(no_transits_were_fit, chi2, chi2red, chi2_min, chi2red_min, test_statistic_periods,
                          test_statistic_depths, transitleastsquares, lc_arr, best_row, period_grid, durations,
                          duration, maxwidth_in_samples)
        results.pop('rp_rs')
        return results

    def __reference_comet_transit(self, t, flux, per):
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
