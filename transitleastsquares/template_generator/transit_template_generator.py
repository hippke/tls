from abc import ABC, abstractmethod

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.interpolation import interp1d


class TransitTemplateGenerator(ABC):
    """
    Root class to be used to implement transit shapes and their behaviours.
    """
    def __init__(self):
        pass

    @abstractmethod
    def reference_transit(self, period_grid, duration_grid, samples, per, rp, a, inc, ecc, w, u, limb_dark):
        """
        Creates a reference transit with the desired shape
        :param period_grid: The grid of periods to be processed
        :param duration_grid: The grid of durations to be processed
        :param samples: The samples count
        :param per: The period for the template
        :param rp: The radius of the planet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param a: The semimajor axis of the planet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param inc: The inclination of the planet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param ecc: The eccentricity of the planet causing the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        :param w:
        :param u:
        :param limb_dark: The limb darkening applied to the transit. This only applies to some templates and is kept to
        respect the original TLS implementation.
        """
        pass

    @abstractmethod
    def duration_grid(self, periods, shortest, log_step=tls_constants.DURATION_GRID_STEP):
        """
        Generates a grid of durations.
        :param periods: The grid of periods to be used.
        :param shortest: The shortest duration.
        :param log_step: The logarithmic step increment factor.
        """
        pass

    @abstractmethod
    def min_duration(self, period, R_star, M_star, periods=None):
        """
        Calculates the minimum duration for this template.
        :param period: The period for which the duration needs to be calculated.
        :param R_star: The radius of the host star.
        :param M_star: The mass of the host star
        :param periods: The period grid.
        """
        pass

    @abstractmethod
    def max_duration(self, period, R_star, M_star, periods=None):
        """
        Calculates the maximum duration for this template.
        :param period: The period for which the duration needs to be calculated.
        :param R_star: The radius of the host star.
        :param M_star: The mass of the host star
        :param periods: The period grid.
        """
        pass

    def fractional_transit(self,
            period_grid,
            duration_grid,
            duration,
            maxwidth,
            depth,
            samples,
            per,
            rp,
            a,
            inc,
            ecc,
            w,
            u,
            limb_dark,
            cached_reference_transit=None,
    ):
        """Returns a scaled reference transit with fractional width and depth"""

        if cached_reference_transit is None:
            reference_flux = self.reference_transit(
                period_grid=period_grid,
                duration_grid=duration_grid,
                samples=samples,
                per=per,
                rp=rp,
                a=a,
                inc=inc,
                ecc=ecc,
                w=w,
                u=u,
                limb_dark=limb_dark,
            )
        else:
            reference_flux = cached_reference_transit

        # Interpolate to shorter interval - new method without scipy
        reference_time = numpy.linspace(-0.5, 0.5, samples)
        occupied_samples = int((duration / maxwidth) * samples)
        x_new = numpy.linspace(-0.5, 0.5, occupied_samples)
        f = interp1d(x_new, reference_time)
        y_new = f(reference_flux)

        # Patch ends with ones ("1")
        missing_samples = samples - occupied_samples
        emtpy_segment = numpy.ones(int(missing_samples * 0.5))
        result = numpy.append(emtpy_segment, y_new)
        result = numpy.append(result, emtpy_segment)
        if numpy.size(result) < samples:  # If odd number of samples
            result = numpy.append(result, numpy.ones(1))

        # Depth rescaling
        result = 1 - ((1 - result) * depth)

        return result

    def get_cache(self, period_grid, durations, maxwidth_in_samples, per, rp, a, inc, ecc, w, u, limb_dark):
        """Fetches (size(durations)*size(depths)) light curves of length
            maxwidth_in_samples and returns these LCs in a 2D array, together with
            their metadata in a separate array."""

        print("Creating model cache for", str(len(durations)), "durations")
        lc_arr = []
        rows = numpy.size(durations)
        lc_cache_overview = numpy.zeros(
            rows,
            dtype=[("duration", "f8"), ("width_in_samples", "i8"), ("overshoot", "f8")],
        )
        cached_reference_transit = self.reference_transit(
            period_grid=period_grid,
            duration_grid=durations,
            samples=maxwidth_in_samples,
            per=per,
            rp=rp,
            a=a,
            inc=inc,
            ecc=ecc,
            w=w,
            u=u,
            limb_dark=limb_dark,
        )

        row = 0
        for duration in durations:
            scaled_transit = self.fractional_transit(
                period_grid=period_grid,
                duration_grid=durations,
                duration=duration,
                maxwidth=numpy.max(durations),
                depth=tls_constants.SIGNAL_DEPTH,
                samples=maxwidth_in_samples,
                per=per,
                rp=rp,
                a=a,
                inc=inc,
                ecc=ecc,
                w=w,
                u=u,
                limb_dark=limb_dark,
                cached_reference_transit=cached_reference_transit,
            )
            lc_cache_overview["duration"][row] = duration
            used_samples = int((duration / numpy.max(durations)) * maxwidth_in_samples)
            lc_cache_overview["width_in_samples"][row] = used_samples
            full_values = numpy.where(
                scaled_transit < (1 - tls_constants.NUMERICAL_STABILITY_CUTOFF)
            )
            first_sample = numpy.min(full_values)
            last_sample = numpy.max(full_values) + 1
            signal = scaled_transit[first_sample:last_sample]
            lc_arr.append(signal)

            # Fraction of transit bottom and mean flux
            overshoot = numpy.mean(signal) / numpy.min(signal)

            # Later, we multiply the inverse fraction ==> convert to inverse percentage
            lc_cache_overview["overshoot"][row] = 1 / (2 - overshoot)
            row += +1

        lc_arr = numpy.array(lc_arr)
        return lc_cache_overview, lc_arr