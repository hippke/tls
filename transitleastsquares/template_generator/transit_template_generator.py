import warnings
from abc import ABC, abstractmethod

import numpy
from numpy import pi, sqrt
from .. import tls_constants
from ..interpolation import interp1d
from tqdm import tqdm
from ..core import fold
from ..results import transitleastsquaresresults
from ..stats import count_stats, snr_stats, model_lightcurve, calculate_stretch, \
    calculate_fill_factor, calculate_transit_duration_in_days, all_transit_times, spectra, intransit_stats, FAP, \
    period_uncertainty, rp_rs_from_depth


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

    def period_grid(self, R_star, M_star, time_span, period_min=0, period_max=float("inf"),
            oversampling_factor=tls_constants.OVERSAMPLING_FACTOR, n_transits_min=tls_constants.N_TRANSITS_MIN):
        """
        Generates a grid of optimal sampling periods for transit search in light curves.
        Following Ofir (2014, A&A, 561, A138)
        :param R_star: The star radius
        :param M_star: The star mass
        :param time_span: The light curve timespam
        :param period_min: The minimum period of the grid
        :param period_max: The maximum period of the grid
        :param oversampling_factor: The grid density free parameter
        :param n_transits_min: The minimum number of transits to be matched
        """
        if R_star < 0.1:
            text = (
                    "Warning: R_star was set to 0.1 for period_grid (was unphysical: "
                    + str(R_star)
                    + ")"
            )
            warnings.warn(text)
            R_star = 0.1

        if R_star > 10000:
            text = (
                    "Warning: R_star was set to 10000 for period_grid (was unphysical: "
                    + str(R_star)
                    + ")"
            )
            warnings.warn(text)
            R_star = 10000

        if M_star < 0.01:
            text = (
                    "Warning: M_star was set to 0.01 for period_grid (was unphysical: "
                    + str(M_star)
                    + ")"
            )
            warnings.warn(text)
            M_star = 0.01

        if M_star > 1000:
            text = (
                    "Warning: M_star was set to 1000 for period_grid (was unphysical: "
                    + str(M_star)
                    + ")"
            )
            warnings.warn(text)
            M_star = 1000

        R_star = R_star * tls_constants.R_sun
        M_star = M_star * tls_constants.M_sun
        time_span = time_span * tls_constants.SECONDS_PER_DAY  # seconds

        # boundary conditions
        f_min = n_transits_min / time_span
        f_max = 1.0 / (2 * pi) * sqrt(tls_constants.G * M_star / (3 * R_star) ** 3)

        # optimal frequency sampling, Equations (5), (6), (7)
        A = (
                (2 * pi) ** (2.0 / 3)
                / pi
                * R_star
                / (tls_constants.G * M_star) ** (1.0 / 3)
                / (time_span * oversampling_factor)
        )
        C = f_min ** (1.0 / 3) - A / 3.0
        N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

        X = numpy.arange(N_opt) + 1
        f_x = (A / 3 * X + C) ** 3
        P_x = 1 / f_x

        # Cut to given (optional) selection of periods
        periods = P_x / tls_constants.SECONDS_PER_DAY
        selected_index = numpy.where(
            numpy.logical_and(periods > period_min, periods <= period_max)
        )

        number_of_periods = numpy.size(periods[selected_index])

        if number_of_periods > 10 ** 6:
            text = (
                    "period_grid generates a very large grid ("
                    + str(number_of_periods)
                    + "). Recommend to check physical plausibility for stellar mass, radius, and time series duration."
            )
            warnings.warn(text)

        if number_of_periods < tls_constants.MINIMUM_PERIOD_GRID_SIZE:
            if time_span < 5 * tls_constants.SECONDS_PER_DAY:
                time_span = 5 * tls_constants.SECONDS_PER_DAY
            warnings.warn(
                "period_grid defaults to R_star=1 and M_star=1 as given density yielded grid with too few values"
            )
            return self.period_grid(
                R_star=1, M_star=1, time_span=time_span / tls_constants.SECONDS_PER_DAY
            )
        else:
            return periods[selected_index]  # periods in [days]

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

    @abstractmethod
    def final_T0_fit(self, signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar):
        """ After the search, we know the best period, width and duration.
            But T0 was not preserved due to speed optimizations.
            Thus, iterate over T0s using the given parameters
            Fold to all T0s so that the transit is expected at phase = 0"""
        pass

    @abstractmethod
    def transit_mask(self, t, period, duration, T0):
        """
        Computes a boolean mask for the entire time set given by 't' and the transit parameters 'duration', 'T0' and
        'period'.
        :param t: the time set
        :param period: the transit period
        :param duration: the transit duration
        :param T0: the transit T0
        :return: the boolean transit mask
        """
        # Works with numba, but is not faster
        mask = numpy.abs((t - T0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration
        return mask

    @abstractmethod
    def calculate_results(self, no_transits_were_fit, chi2, chi2red, chi2_min, chi2red_min, test_statistic_periods,
                          test_statistic_depths, transitleastsquares, lc_arr, best_row, period_grid, durations,
                          duration, maxwidth_in_samples):
        """
        Returns a transitleastsquaresresult for the given computed statistics.
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