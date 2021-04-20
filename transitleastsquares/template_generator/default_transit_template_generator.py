import batman
import numpy
import warnings
from .. import tls_constants
from ..grid import T14
from ..interpolation import interp1d
from .transit_template_generator import TransitTemplateGenerator
from tqdm import tqdm
from ..core import fold
from ..results import transitleastsquaresresults
from ..stats import count_stats, snr_stats, model_lightcurve, calculate_stretch, \
    calculate_fill_factor, calculate_transit_duration_in_days, all_transit_times, spectra, intransit_stats, FAP, \
    period_uncertainty, rp_rs_from_depth


class DefaultTransitTemplateGenerator(TransitTemplateGenerator):
    """
    Default implementation used by TLS.
    """
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
        duration_max = self.max_duration(min(periods), tls_constants.R_STAR_MAX, tls_constants.M_STAR_MAX)
        duration_min = self.min_duration(max(periods), tls_constants.R_STAR_MIN, tls_constants.M_STAR_MIN)
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
        return T14(R_s=R_star, M_s=M_star, P=period, small=False)

    def final_T0_fit(self, signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar):
        dur = len(signal)
        scale = tls_constants.SIGNAL_DEPTH / (1 - depth) if depth >= 0 else tls_constants.SIGNAL_DEPTH / (1 + depth)
        signal = [1 - ((1 - value) / scale) if value <= 1 else 1 + ((value - 1) / scale) for value in signal]
        samples_per_period = numpy.size(y)

        if T0_fit_margin == 0:
            points = samples_per_period
        else:
            step_factor = T0_fit_margin * dur
            points = int(samples_per_period / step_factor)
        if points > samples_per_period:
            points = samples_per_period

        # Create all possible T0s from the start of [t] to [t+period] in [samples] steps
        T0_array = numpy.linspace(
            start=numpy.min(t), stop=numpy.min(t) + period, num=points
        )

        # Avoid showing progress bar when expected runtime is short
        if points > tls_constants.PROGRESSBAR_THRESHOLD and show_progress_bar:
            show_progress_info = True
        else:
            show_progress_info = False

        residuals_lowest = float("inf")
        T0 = 0

        if show_progress_info:
            print("Searching for best T0 for period", format(period, ".5f"), "days")
            pbar2 = tqdm(total=numpy.size(T0_array))
        signal_ootr = numpy.ones(len(y[dur:]))

        # Future speed improvement possible: Add multiprocessing. Will be slower for
        # short data and T0_FIT_MARGIN > 0.01, but faster for large data with dense
        # sampling (T0_FIT_MARGIN=0)
        for Tx in T0_array:
            phases = fold(time=t, period=period, T0=Tx)
            sort_index = numpy.argsort(phases, kind="mergesort")  # 75% of CPU time
            phases = phases[sort_index]
            flux = y[sort_index]
            dy = dy[sort_index]

            # Roll so that the signal starts at index 0
            # Numpy roll is slow, so we replace it with less elegant concatenate
            # flux = numpy.roll(flux, roll_cadences)
            # dy = numpy.roll(dy, roll_cadences)
            roll_cadences = int(dur / 2) + 1
            flux = numpy.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])
            dy = numpy.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])

            residuals_intransit = numpy.sum((flux[:dur] - signal) ** 2 / dy[:dur] ** 2)
            residuals_ootr = numpy.sum((flux[dur:] - signal_ootr) ** 2 / dy[dur:] ** 2)
            residuals_total = residuals_intransit + residuals_ootr

            if show_progress_info:
                pbar2.update(1)
            if residuals_total < residuals_lowest:
                residuals_lowest = residuals_total
                T0 = Tx
        if show_progress_info:
            pbar2.close()
        return T0

    def transit_mask(self, t, period, duration, T0):
        # Works with numba, but is not faster
        mask = numpy.abs((t - T0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration
        return mask

    def calculate_results(self, no_transits_were_fit, chi2, chi2red, chi2_min, chi2red_min, test_statistic_periods,
                          test_statistic_depths, transitleastsquares, lc_arr, best_row, period_grid, durations,
                          duration, maxwidth_in_samples):
        """
        Returns a transitleastsquaresresult for the given template
        """
        if no_transits_were_fit:
            power_raw = numpy.zeros(len(chi2))
            power = numpy.zeros(len(chi2))
            period = numpy.nan
            depth = 1
            SR = 0
            SDE = 0
            SDE_raw = 0
            T0 = 0
            transit_times = numpy.nan
            transit_duration_in_days = numpy.nan
            internal_samples = (
                int(len(transitleastsquares.y)) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE
            )
            folded_phase = numpy.nan
            folded_y = numpy.nan
            folded_dy = numpy.nan
            model_folded_phase = numpy.nan
            model_folded_model = numpy.nan
            model_transit_single = numpy.nan
            model_lightcurve_model = numpy.nan
            model_lightcurve_time = numpy.nan
            depth_mean_odd = numpy.nan
            depth_mean_even = numpy.nan
            depth_mean_odd_std = numpy.nan
            depth_mean_even_std = numpy.nan
            all_flux_intransit_odd = numpy.nan
            all_flux_intransit_even = numpy.nan
            per_transit_count = numpy.nan
            transit_depths = numpy.nan
            transit_depths_uncertainties = numpy.nan
            all_flux_intransit = numpy.nan
            snr_per_transit = numpy.nan
            snr_pink_per_transit = numpy.nan
            depth_mean = numpy.nan
            depth_mean_std = numpy.nan
            snr = numpy.nan
            rp_rs = numpy.nan
            depth_mean_odd = numpy.nan
            depth_mean_even = numpy.nan
            depth_mean_odd_std = numpy.nan
            depth_mean_even_std = numpy.nan
            odd_even_difference = numpy.nan
            odd_even_std_sum = numpy.nan
            odd_even_mismatch = numpy.nan
            transit_count = numpy.nan
            empty_transit_count = numpy.nan
            distinct_transit_count = numpy.nan
            duration = numpy.nan
            in_transit_count = numpy.nan
            after_transit_count = numpy.nan
            before_transit_count = numpy.nan
        else:
            SR, power_raw, power, SDE_raw, SDE = spectra(chi2, transitleastsquares.oversampling_factor)
            index_highest_power = numpy.argmax(power)
            period = test_statistic_periods[index_highest_power]
            depth = test_statistic_depths[index_highest_power]
            T0 = self.final_T0_fit(
                signal=lc_arr[best_row],
                depth=depth,
                t=transitleastsquares.t,
                y=transitleastsquares.y,
                dy=transitleastsquares.dy,
                period=period,
                T0_fit_margin=transitleastsquares.T0_fit_margin,
                show_progress_bar=transitleastsquares.show_progress_bar,
            )
            transit_times = all_transit_times(T0, transitleastsquares.t, period)

            transit_duration_in_days = calculate_transit_duration_in_days(
                transitleastsquares.t, period, transit_times, duration
            )
            phases = fold(transitleastsquares.t, period, T0=T0 + period / 2)
            sort_index = numpy.argsort(phases)
            folded_phase = phases[sort_index]
            folded_y = transitleastsquares.y[sort_index]
            folded_dy = transitleastsquares.dy[sort_index]
            # Model phase, shifted by half a cadence so that mid-transit is at phase=0.5
            model_folded_phase = numpy.linspace(
                0 + 1 / numpy.size(transitleastsquares.t) / 2,
                1 + 1 / numpy.size(transitleastsquares.t) / 2,
                numpy.size(transitleastsquares.t),
            )
            # Folded model / model curve
            # Data phase 0.5 is not always at the midpoint (not at cadence: len(y)/2),
            # so we need to roll the model to match the model so that its mid-transit
            # is at phase=0.5
            fill_factor = calculate_fill_factor(transitleastsquares.t)
            fill_half = 1 - ((1 - fill_factor) * 0.5)
            stretch = calculate_stretch(transitleastsquares.t, period, transit_times)
            internal_samples = (
                int(len(transitleastsquares.y) / len(transit_times))
            ) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE

            # Folded model flux
            model_folded_model = self.fractional_transit(
                period_grid=period_grid,
                duration_grid=durations,
                duration=duration * maxwidth_in_samples * fill_half,
                maxwidth=maxwidth_in_samples / stretch,
                depth=1 - depth,
                samples=int(len(transitleastsquares.t / len(transit_times))),
                per=transitleastsquares.per,
                rp=transitleastsquares.rp,
                a=transitleastsquares.a,
                inc=transitleastsquares.inc,
                ecc=transitleastsquares.ecc,
                w=transitleastsquares.w,
                u=transitleastsquares.u,
                limb_dark=transitleastsquares.limb_dark,
            )
            # Full unfolded light curve model
            model_transit_single = self.fractional_transit(
                period_grid=period_grid,
                duration_grid=durations,
                duration=(duration * maxwidth_in_samples),
                maxwidth=maxwidth_in_samples / stretch,
                depth=1 - depth,
                samples=internal_samples,
                per=transitleastsquares.per,
                rp=transitleastsquares.rp,
                a=transitleastsquares.a,
                inc=transitleastsquares.inc,
                ecc=transitleastsquares.ecc,
                w=transitleastsquares.w,
                u=transitleastsquares.u,
                limb_dark=transitleastsquares.limb_dark,
            )
            model_lightcurve_model, model_lightcurve_time = model_lightcurve(
                transit_times, period, transitleastsquares.t, model_transit_single
            )
            depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, \
            all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
                transitleastsquares.t, transitleastsquares.y, transit_times, transit_duration_in_days
            )
            all_flux_intransit = numpy.concatenate(
                [all_flux_intransit_odd, all_flux_intransit_even]
            )
            snr_per_transit, snr_pink_per_transit = snr_stats(
                t=transitleastsquares.t,
                y=transitleastsquares.y,
                period=period,
                duration=duration,
                T0=T0,
                transit_times=transit_times,
                transit_duration_in_days=transit_duration_in_days,
                per_transit_count=per_transit_count,
                intransit=self.transit_mask(transitleastsquares.t, period, 2 * duration, T0)
            )
            intransit = self.transit_mask(transitleastsquares.t, period, 2 * duration, T0)
            flux_ootr = transitleastsquares.y[~intransit]
            depth_mean = numpy.mean(all_flux_intransit)
            depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
                per_transit_count
            ) ** (0.5)
            flux_depth_mean_diff = 1 - depth_mean if depth_mean < 1 else depth_mean - 1
            snr = (flux_depth_mean_diff / numpy.std(flux_ootr)) * len(
                all_flux_intransit
            ) ** (0.5)

            if len(all_flux_intransit_odd) > 0:
                depth_mean_odd = numpy.mean(all_flux_intransit_odd)
                depth_mean_odd_std = numpy.std(all_flux_intransit_odd) / numpy.sum(
                    len(all_flux_intransit_odd)
                ) ** (0.5)
            else:
                depth_mean_odd = numpy.nan
                depth_mean_odd_std = numpy.nan

            if len(all_flux_intransit_even) > 0:
                depth_mean_even = numpy.mean(all_flux_intransit_even)
                depth_mean_even_std = numpy.std(all_flux_intransit_even) / numpy.sum(
                    len(all_flux_intransit_even)
                ) ** (0.5)
            else:
                depth_mean_even = numpy.nan
                depth_mean_even_std = numpy.nan

            in_transit_count, after_transit_count, before_transit_count = count_stats(
                transitleastsquares.t, transitleastsquares.y, transit_times, transit_duration_in_days
            )

            # Odd even mismatch in standard deviations
            odd_even_difference = abs(depth_mean_odd - depth_mean_even)
            odd_even_std_sum = depth_mean_odd_std + depth_mean_even_std
            odd_even_mismatch = odd_even_difference / odd_even_std_sum

            transit_count = len(transit_times)
            empty_transit_count = numpy.count_nonzero(per_transit_count == 0)
            distinct_transit_count = transit_count - empty_transit_count

            duration = transit_duration_in_days

            if empty_transit_count / transit_count >= 0.33:
                text = (
                    str(empty_transit_count)
                    + " of "
                    + str(transit_count)
                    + " transits without data. The true period may be twice the given period."
                )
                warnings.warn(text)

            rp_rs = rp_rs_from_depth(depth=1 - depth, law=transitleastsquares.limb_dark, params=transitleastsquares.u)
        return transitleastsquaresresults(
            SDE,
            SDE_raw,
            chi2_min,
            chi2red_min,
            period,
            period_uncertainty(test_statistic_periods, power),
            T0,
            duration,
            depth,
            (depth_mean, depth_mean_std),
            (depth_mean_even, depth_mean_even_std),
            (depth_mean_odd, depth_mean_odd_std),
            transit_depths,
            transit_depths_uncertainties,
            rp_rs,
            snr,
            snr_per_transit,
            snr_pink_per_transit,
            odd_even_mismatch,
            transit_times,
            per_transit_count,
            transit_count,
            distinct_transit_count,
            empty_transit_count,
            FAP(SDE),
            in_transit_count,
            after_transit_count,
            before_transit_count,
            test_statistic_periods,
            power,
            power_raw,
            SR,
            chi2,
            chi2red,
            model_lightcurve_time,
            model_lightcurve_model,
            model_folded_phase,
            folded_y,
            folded_dy,
            folded_phase,
            model_folded_model,
        )
