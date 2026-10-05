"""Result container of a TLS search."""

RESULT_KEYS = (
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
    "in_transit_count",
    "after_transit_count",
    "before_transit_count",
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
)


class transitleastsquaresresults(dict):
    """The results of a transitleastsquares search: a dict whose items are also
    accessible as attributes (results.period == results["period"]).

    Construct either positionally (values in RESULT_KEYS order, as in TLS <= 1.33)
    or with keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(zip(RESULT_KEYS, args))
        else:
            super().__init__((k, kwargs[k]) for k in RESULT_KEYS if k in kwargs)
            self.update((k, v) for k, v in kwargs.items() if k not in RESULT_KEYS)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
