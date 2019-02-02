#  Optimized algorithm to search for transits of small extrasolar planets
#                                                                            /
#       ,        AUTHORS                                                   O/
#    \  :  /     Michael Hippke (1) [michael@hippke.org]                /\/|
# `. __/ \__ .'  Rene' Heller (2) [heller@mps.mpg.de]                      |
# _ _\     /_ _  _________________________________________________________/ \_
#    /_   _\
#  .'  \ /  `.   (1) Sonneberg Observatory, Sternwartestr. 32, Sonneberg
#    /  :  \     (2) Max Planck Institute for Solar System Research,
#       '            Justus-von-Liebig-Weg 3, 37077 G\"ottingen, Germany

from os import path
import multiprocessing
import numpy
import sys
import warnings
import configparser
from functools import partial
from tqdm import tqdm

# TLS parts
import transitleastsquares.tls_constants as tls_constants
from transitleastsquares.stats import (
    FAP,
    rp_rs_from_depth,
    pink_noise,
    period_uncertainty,
    spectra,
    final_T0_fit,
    model_lightcurve,
    all_transit_times,
    calculate_transit_duration_in_days,
    calculate_stretch,
    calculate_fill_factor,
    intransit_stats
    )
from transitleastsquares.catalog import catalog_info
from transitleastsquares.helpers import (
    resample,
    cleaned_array,
    transit_mask,
    )
from transitleastsquares.helpers import impact_to_inclination
from transitleastsquares.grid import (
    duration_grid,
    period_grid
    )
from transitleastsquares.core import (
    edge_effect_correction,
    lowest_residuals_in_this_duration,
    out_of_transit_residuals,
    fold,
    foldfast,
    search_period
    )
from transitleastsquares.transit import reference_transit, fractional_transit, get_cache

numpy.warnings.filterwarnings('ignore')



class transitleastsquares(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy)

    def _validate_inputs(self, t, y, dy):
        """Check the consistency of the inputs"""

        # Clean array
        if dy is None:
            t, y = cleaned_array(t, y)
        else:
            t, y, dy = cleaned_array(t, y, dy)
            # Normalize dy to act as weights in least squares calculatio
            dy = dy / numpy.mean(dy)

        duration = max(t) - min(t)
        if duration <= 0:
            raise ValueError("Time duration must positive")
        if numpy.size(y) < 3 or numpy.size(t) < 3:
            raise ValueError("Too few values in data set")
        if numpy.mean(y) > 1.01 or numpy.mean(y) < 0.99:
            warnings.warn(
                "Warning: The mean flux should be normalized to 1"
                + ", but it was found to be "
                + str(numpy.mean(y))
            )
        if min(y) < 0:
            raise ValueError("Flux values must be positive")
        if max(y) >= float("inf"):
            raise ValueError("Flux values must be finite")

        # If no dy is given, create it with the standard deviation of the flux
        if dy is None:
            dy = numpy.full(len(y), numpy.std(y))
        if numpy.size(t) != numpy.size(y) or numpy.size(t) != numpy.size(dy):
            raise ValueError("Arrays (t, y, dy) must be of the same dimensions")
        if t.ndim != 1:  # Size identity ensures dimensional identity
            raise ValueError("Inputs (t, y, dy) must be 1-dimensional")

        return t, y, dy

    

    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""

        print(tls_constants.TLS_VERSION)

        # Validate **kwargs and set to defaults where missing
        self.transit_depth_min = kwargs.get(
            "transit_depth_min", tls_constants.TRANSIT_DEPTH_MIN
        )
        self.R_star = kwargs.get("R_star", tls_constants.R_STAR)
        self.M_star = kwargs.get("M_star", tls_constants.M_STAR)
        self.oversampling_factor = kwargs.get(
            "oversampling_factor", tls_constants.OVERSAMPLING_FACTOR
        )
        self.period_max = kwargs.get("period_max", float("inf"))
        self.period_min = kwargs.get("period_min", 0)
        self.n_transits_min = kwargs.get("n_transits_min", tls_constants.N_TRANSITS_MIN)

        self.R_star_min = kwargs.get("R_star_min", tls_constants.R_STAR_MIN)
        self.R_star_max = kwargs.get("R_star_max", tls_constants.R_STAR_MAX)
        self.M_star_min = kwargs.get("M_star_min", tls_constants.M_STAR_MIN)
        self.M_star_max = kwargs.get("M_star_max", tls_constants.M_STAR_MAX)
        self.duration_grid_step = kwargs.get(
            "duration_grid_step", tls_constants.DURATION_GRID_STEP
        )

        self.use_threads = kwargs.get("use_threads", multiprocessing.cpu_count())

        self.per = kwargs.get("per", tls_constants.DEFAULT_PERIOD)
        self.rp = kwargs.get("rp", tls_constants.DEFAULT_RP)
        self.a = kwargs.get("a", tls_constants.DEFAULT_A)

        self.T0_fit_margin = kwargs.get("T0_fit_margin", tls_constants.T0_FIT_MARGIN)

        # If an impact parameter is given, it overrules the supplied inclination
        if "b" in kwargs:
            self.b = kwargs.get("b")
            self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)
        else:
            self.inc = kwargs.get("inc", tls_constants.DEFAULT_INC)

        self.ecc = kwargs.get("ecc", tls_constants.DEFAULT_ECC)
        self.w = kwargs.get("w", tls_constants.DEFAULT_W)
        self.u = kwargs.get("u", tls_constants.DEFAULT_U)
        self.limb_dark = kwargs.get("limb_dark", tls_constants.DEFAULT_LIMB_DARK)

        self.transit_template = kwargs.get("transit_template", "default")
        if self.transit_template == "default":
            self.per = tls_constants.DEFAULT_PERIOD
            self.rp = tls_constants.DEFAULT_RP
            self.a = tls_constants.DEFAULT_A
            self.inc = tls_constants.DEFAULT_INC

        elif self.transit_template == "grazing":
            self.b = tls_constants.GRAZING_B
            self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)

        elif self.transit_template == "box":
            self.per = tls_constants.BOX_PERIOD
            self.rp = tls_constants.BOX_RP
            self.a = tls_constants.BOX_A
            self.b = tls_constants.BOX_B
            self.inc = tls_constants.BOX_INC
            self.u = tls_constants.BOX_U
            self.limb_dark = tls_constants.BOX_LIMB_DARK

        else:
            raise ValueError(
                'Unknown transit_template. Known values: \
                "default", "grazing", "box"'
            )

        """Validations to avoid (garbage in ==> garbage out)"""

        # Stellar radius
        # 0 < R_star < inf
        if self.R_star <= 0 or self.R_star >= float("inf"):
            raise ValueError("R_star must be positive")

        # Assert (0 < R_star_min <= R_star)
        if self.R_star_min > self.R_star:
            raise ValueError("R_star_min <= R_star is required")
        if self.R_star_min <= 0 or self.R_star_min >= float("inf"):
            raise ValueError("R_star_min must be positive")

        # Assert (R_star <= R_star_max < inf)
        if self.R_star_max < self.R_star:
            raise ValueError("R_star_max >= R_star is required")
        if self.R_star_max <= 0 or self.R_star_max >= float("inf"):
            raise ValueError("R_star_max must be positive")

        # Stellar mass
        # Assert (0 < M_star < inf)
        if self.M_star <= 0 or self.M_star >= float("inf"):
            raise ValueError("M_star must be positive")

        # Assert (0 < M_star_min <= M_star)
        if self.M_star_min > self.M_star:
            raise ValueError("M_star_min <= M_star is required")
        if self.M_star_min <= 0 or self.M_star_min >= float("inf"):
            raise ValueError("M_star_min must be positive")

        # Assert (M_star <= M_star_max < inf)
        if self.M_star_max < self.M_star:
            raise ValueError("M_star_max >= M_star required")
        if self.M_star_max <= 0 or self.M_star_max >= float("inf"):
            raise ValueError("M_star_max must be positive")

        # Period grid
        if self.period_min < 0:
            raise ValueError("period_min >= 0 required")
        if self.period_min >= self.period_max:
            raise ValueError("period_min < period_max required")
        if not isinstance(self.n_transits_min, int):
            raise ValueError("n_transits_min must be an integer value")
        if self.n_transits_min < 1:
            raise ValueError("n_transits_min must be an integer value >= 1")

        if not isinstance(self.use_threads, int) or self.use_threads < 1:
            raise ValueError("use_threads must be an integer value >= 1")

        # Assert 0 < T0_fit_margin < 0.1
        if self.T0_fit_margin < 0:
            self.T0_fit_margin = 0
        elif self.T0_fit_margin > 0.1:  # Sensible limit 10% of transit duration
            self.T0_fit_margin = 0.1

        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t) - numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        durations = duration_grid(
            periods, shortest=1 / len(self.t), log_step=self.duration_grid_step
        )

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

        # Result lists now (faster), convert to numpy array later
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rows = []
        test_statistic_depths = []

        print(
            "Searching "
            + str(len(self.y))
            + " data points, "
            + str(len(periods))
            + " periods from "
            + str(round(min(periods), 3))
            + " to "
            + str(round(max(periods), 3))
            + " days"
        )
        if self.use_threads == multiprocessing.cpu_count():
            print("Using all " + str(self.use_threads) + " CPU threads")
        else:
            print(
                "Using "
                + str(self.use_threads)
                + " of "
                + str(multiprocessing.cpu_count())
                + " CPU threads"
            )

        p = multiprocessing.Pool(processes=self.use_threads)
        params = partial(
            search_period,
            t=self.t,
            y=self.y,
            dy=self.dy,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
            lc_arr=lc_arr,
            lc_cache_overview=lc_cache_overview,
            T0_fit_margin=self.T0_fit_margin,
        )
        bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}"
        pbar = tqdm(total=numpy.size(periods), smoothing=0.3, bar_format=bar_format)

        if tls_constants.PERIODS_SEARCH_ORDER == "ascending":
            periods = reversed(periods)
        elif tls_constants.PERIODS_SEARCH_ORDER == "descending":
            pass  # it already is
        elif tls_constants.PERIODS_SEARCH_ORDER == "shuffled":
            periods = numpy.random.permutation(periods)
        else:
            raise ValueError("Unknown PERIODS_SEARCH_ORDER")

        for data in p.imap_unordered(params, periods):
            test_statistic_periods.append(data[0])
            test_statistic_residuals.append(data[1])
            test_statistic_rows.append(data[2])
            test_statistic_depths.append(data[3])
            pbar.update(1)
        p.close()
        pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(test_statistic_periods)
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[sort_index]
        test_statistic_residuals = numpy.array(test_statistic_residuals)[sort_index]
        test_statistic_rows = numpy.array(test_statistic_rows)[sort_index]
        test_statistic_depths = numpy.array(test_statistic_depths)[sort_index]

        idx_best = numpy.argmin(test_statistic_residuals)
        best_row = test_statistic_rows[idx_best]
        duration = lc_cache_overview["duration"][best_row]
        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))

        if max(test_statistic_residuals) == min(test_statistic_residuals):
            no_transits_were_fit = True
            warnings.warn('No transit were fit. Try smaller "transit_depth_min"')
        else:
            no_transits_were_fit = False

        # Power spectra variants
        chi2 = test_statistic_residuals
        degrees_of_freedom = 4
        chi2red = test_statistic_residuals / (len(self.t) - degrees_of_freedom)
        chi2_min = numpy.min(chi2)
        chi2red_min = numpy.min(chi2red)

        if no_transits_were_fit:
            power_raw = numpy.zeros(len(chi2))
            power = numpy.zeros(len(chi2))
            period = numpy.nan
            depth = 1
            SR = 0
            SDE = 0
            SDE_raw = 0
        else:
            SR, power_raw, power, SDE_raw, SDE = spectra(chi2, self.oversampling_factor)
            index_highest_power = numpy.argmax(power)
            period = test_statistic_periods[index_highest_power]
            depth = test_statistic_depths[index_highest_power]

        if no_transits_were_fit:
            T0 = 0
        else:
            T0 = final_T0_fit(
                signal=lc_arr[best_row],
                depth=depth,
                t=self.t,
                y=self.y,
                dy=self.dy,
                period=period,
                T0_fit_margin=self.T0_fit_margin
                )

        transit_times = all_transit_times(T0, self.t, period)

        transit_duration_in_days =  calculate_transit_duration_in_days(
            self.t,
            period,
            transit_times,
            duration
            )

        phases = fold(self.t, period, T0=T0 + period / 2)
        sort_index = numpy.argsort(phases)
        folded_phase = phases[sort_index]
        folded_y = self.y[sort_index]
        folded_dy = self.dy[sort_index]

        # Model phase, shifted by half a cadence so that mid-transit is at phase=0.5
        model_folded_phase = numpy.linspace(
            0 + 1 / numpy.size(phases) / 2,
            1 + 1 / numpy.size(phases) / 2,
            numpy.size(phases),
        )

        # Folded model / model curve
        # Data phase 0.5 is not always at the midpoint (not at cadence: len(y)/2),
        # so we need to roll the model to match the model so that its mid-transit
        # is at phase=0.5
        fill_factor = calculate_fill_factor(self.t)
        fill_half = 1 - ((1 - fill_factor) * 0.5)

        stretch = calculate_stretch(self.t, period, transit_times)

        # Folded model flux
        if no_transits_were_fit:
            model_folded_model = numpy.ones(len(model_folded_phase))
        else:
            model_folded_model = fractional_transit(
                duration=duration * maxwidth_in_samples * fill_half,
                maxwidth=maxwidth_in_samples / stretch,
                depth=1 - depth,
                samples=int(len(self.t / len(transit_times))),
                per=self.per,
                rp=self.rp,
                a=self.a,
                inc=self.inc,
                ecc=self.ecc,
                w=self.w,
                u=self.u,
                limb_dark=self.limb_dark,
            )

        # Full unfolded light curve model
        internal_samples = (int(len(self.y) / len(transit_times))) * \
            tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE
        if no_transits_were_fit:
            model_transit_single = numpy.ones(internal_samples)
        else:
            model_transit_single = fractional_transit(
                duration=(duration * maxwidth_in_samples),
                maxwidth=maxwidth_in_samples / stretch,
                depth=1 - depth,
                samples=internal_samples,
                per=self.per,
                rp=self.rp,
                a=self.a,
                inc=self.inc,
                ecc=self.ecc,
                w=self.w,
                u=self.u,
                limb_dark=self.limb_dark,
            )
        model_lightcurve_model, model_lightcurve_time = model_lightcurve(
            transit_times,
            period,
            self.t,
            model_transit_single
            )

        depth_mean_odd, \
        depth_mean_even, \
        depth_mean_odd_std, \
        depth_mean_even_std, \
        all_flux_intransit_odd, \
        all_flux_intransit_even, \
        per_transit_count,\
        transit_depths, \
        transit_depths_uncertainties = intransit_stats(
                self.t, self.y, transit_times, transit_duration_in_days)
        
        snr_per_transit = numpy.zeros([len(transit_times)])
        snr_pink_per_transit = numpy.zeros([len(transit_times)])
        all_flux_intransit = numpy.array([])
        all_idx_intransit = numpy.array([])

        flux_ootr = numpy.delete(self.y, all_idx_intransit.astype(int))

        # Estimate SNR and pink SNR
        # Second run because now the out of transit points are known
        std = numpy.std(flux_ootr)
        for i in range(len(transit_times)):
            mid_transit = transit_times[i]
            tmin = mid_transit - 0.5 * transit_duration_in_days
            tmax = mid_transit + 0.5 * transit_duration_in_days
            idx_intransit = numpy.where(numpy.logical_and(self.t > tmin, self.t < tmax))
            all_idx_intransit = numpy.append(all_idx_intransit, idx_intransit)
            flux_intransit = self.y[idx_intransit]
            all_flux_intransit = numpy.append(all_flux_intransit, flux_intransit)
            mean_flux = numpy.mean(self.y[idx_intransit])
            intransit_points = numpy.size(self.y[idx_intransit])
            try:
                pinknoise = pink_noise(flux_ootr, int(numpy.mean(per_transit_count)))
                snr_pink_per_transit[i] = (1 - mean_flux) / pinknoise
                std_binned = std / intransit_points ** 0.5
                snr_per_transit[i] = (1 - mean_flux) / std_binned
            except:
                snr_per_transit[i] = 0
                snr_pink_per_transit[i] = 0

        depth_mean = numpy.mean(all_flux_intransit)
        depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
            per_transit_count
        ) ** (0.5)
        snr = ((1 - depth_mean) / numpy.std(flux_ootr)) * len(all_flux_intransit) ** (
            0.5
        )

        rp_rs = rp_rs_from_depth(depth=1 - depth, law=self.limb_dark, params=self.u)

        depth_mean_odd = numpy.mean(all_flux_intransit_odd)
        depth_mean_even = numpy.mean(all_flux_intransit_even)
        depth_mean_odd_std = numpy.std(all_flux_intransit_odd) / numpy.sum(
            len(all_flux_intransit_odd)
        ) ** (0.5)
        depth_mean_even_std = numpy.std(all_flux_intransit_even) / numpy.sum(
            len(all_flux_intransit_even)
        ) ** (0.5)

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


class transitleastsquaresresults(dict):
    """The results of a TransitLeastSquares search"""

    def __init__(self, *args):
        super(transitleastsquaresresults, self).__init__(
            zip(
                (
                    "SDE",
                    "SDE_raw",
                    "chi2_min",
                    "chi2red_min",
                    "period",
                    "period_uncertainty",
                    "T0",
                    "duration",
                    "depth",
                    "depth_mean",
                    "depth_mean_even",
                    "depth_mean_odd",
                    "transit_depths",
                    "transit_depths_uncertainties",
                    "rp_rs",
                    "snr",
                    "snr_per_transit",
                    "snr_pink_per_transit",
                    "odd_even_mismatch",
                    "transit_times",
                    "per_transit_count",
                    "transit_count",
                    "distinct_transit_count",
                    "empty_transit_count",
                    "FAP",
                    "periods",
                    "power",
                    "power_raw",
                    "SR",
                    "chi2",
                    "chi2red",
                    "model_lightcurve_time",
                    "model_lightcurve_model",
                    "model_folded_phase",
                    "folded_y",
                    "folded_dy",
                    "folded_phase",
                    "model_folded_model",
                ),
                args,
            )
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# This is the command line interface
if __name__ == "__main__":
    try:
        import argparse
    except:
        raise ImportError("Could not import package argparse")

    parser = argparse.ArgumentParser()
    parser.add_argument("lightcurve", help="path to lightcurve file")
    parser.add_argument("-o", "--output", help="path to output directory")
    parser.add_argument("-c", "--config", help="path to configuration file")
    args = parser.parse_args()

    use_config_file = False
    if args.config is not None:
        try:
            config = configparser.ConfigParser()
            config.read(args.config)
            R_star = float(config["Grid"]["R_star"])
            R_star_min = float(config["Grid"]["R_star_min"])
            R_star_max = float(config["Grid"]["R_star_max"])
            M_star = float(config["Grid"]["M_star"])
            M_star_min = float(config["Grid"]["M_star_min"])
            M_star_max = float(config["Grid"]["M_star_max"])
            period_min = float(config["Grid"]["period_min"])
            period_max = float(config["Grid"]["period_max"])
            n_transits_min = int(config["Grid"]["n_transits_min"])
            transit_template = config["Template"]["transit_template"]
            duration_grid_step = float(config["Speed"]["duration_grid_step"])
            transit_depth_min = float(config["Speed"]["transit_depth_min"])
            oversampling_factor = int(config["Speed"]["oversampling_factor"])
            T0_fit_margin = int(config["Speed"]["T0_fit_margin"])
            use_config_file = True
            print("Using TLS configuration from config file", args.config)
        except:
            print(
                "Using default values because of broken or missing configuration file",
                args.config,
            )
    else:
        print("No config file given. Using default values")

    data = numpy.genfromtxt(args.lightcurve)
    time = data[:, 0]
    flux = data[:, 1]

    try:
        dy = data[:, 2]
    except:
        dy = numpy.full(len(flux), numpy.std(flux))

    time, flux, dy = cleaned_array(time, flux, dy)

    model = transitleastsquares(time, flux, dy)

    if use_config_file:
        results = model.power(
            R_star=R_star,
            R_star_min=R_star_min,
            R_star_max=R_star_max,
            M_star=M_star,
            M_star_min=M_star_min,
            M_star_max=M_star_max,
            period_min=period_min,
            period_max=period_max,
            n_transits_min=n_transits_min,
            transit_template=transit_template,
            duration_grid_step=duration_grid_step,
            transit_depth_min=transit_depth_min,
            oversampling_factor=oversampling_factor,
            T0_fit_margin=T0_fit_margin,
        )
    else:
        results = model.power()

    if args.output is None:
        target_path_file = "TLS_results.csv"
    else:
        target_path_file = args.output

    try:
        numpy.set_printoptions(precision=8, threshold=10e10)
        with open(target_path_file, "w") as f:
            for key in results.keys():
                f.write("%s %s\n" % (key, results[key]))
        print("Results saved to", target_path_file)
    except IOError:
        print("Error saving result file")