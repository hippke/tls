GP search
=====================================

``transitleastsquares.gp`` searches for periodic transits while modelling the star with a Gaussian process (GP). Instead of detrending first and searching afterwards, the GP is integrated out inside the search: for every trial transit, the depth is fitted jointly with the stellar variability, so that the detrending neither suppresses the transit nor biases its depth. The module is new in TLS 2.1 and experimental; the TLS search itself is unchanged and does not import it.

Quick start::

    from transitleastsquares import gp

    results = gp.search(t, y, dy)                    # CPU: rotation GP, fitted, then the search
    results = gp.search(t, y, dy, device="gpu")      # the same on an NVIDIA GPU
    print(results.period, results.T0, results.duration, results.depth, results.SDE)

``t`` is the time in days, ``y`` the flux normalised to about 1 (not detrended), ``dy`` the flux uncertainties (optional: estimated from the point-to-point scatter). Tutorial 13 shows a full example, tutorial 14 the reproduction of the PLATO benchmark of Talens et al. (2026).

How it works
------------------------

1. The GP hyperparameters are fitted by maximum likelihood (a quasi-Newton method with the exact gradient of the likelihood), by default on data binned to 10 min, with flares removed iteratively (positive outliers; dips are never clipped). The default kernel ``"rotation"`` is the rotation kernel of celerite2 (two damped oscillators at the rotation period and half of it), plus a slow non-periodic term and white-noise jitter. ``"spline"`` is a smoothing spline.
2. A Kalman filter whitens the light curve. For every trial duration and mid-transit time, the single transit event is the :math:`\chi^2` gain of a transit template with a free depth, with the GP marginalised.
3. The single events are summed over the transits of every trial period (on the TLS period grid) and turned into an SDE spectrum as in TLS.

The cost per trial period does not depend on the number of data points, so that the search scales well to long light curves.


Parameters
------------------------

Baseline and fit:

:baseline: *(str or Baseline, default: "rotation")* ``"rotation"`` or ``"spline"`` (fitted by maximum likelihood), or a ``gp.RotationBaseline`` / ``gp.SplineBaseline`` instance, used as given (no fit, no flare clipping).
:clip_flares: *(bool, default: True)* Remove positive outliers iteratively when fitting.
:rotation_period: *(float)* Starting guess of the rotation period for the fit (default: the Lomb-Scargle peak).
:n_starts: *(int, default: 1)* Starting points of the rotation fit (1-3; 1-4 with ``fit_method="nelder-mead"``).
:fit_method: *(str, default: "gradient")* Optimizer of the rotation fit. ``"gradient"``: bounded quasi-Newton with the exact gradient of the likelihood (on the CPU by reverse mode; with ``device="gpu"`` on the GPU, the smoother in segments with single-precision kernels and the likelihood in double precision). ``"nelder-mead"``: the derivative-free simplex, the default of TLS 2.1; it needs about 30 times more likelihood evaluations and mostly reaches lower likelihoods, so its fits, and the search results, differ slightly. On 300 light curves of the Talens et al. (2026) benchmark both found the same planets to within the binomial uncertainty (194 and 195). The spline baseline is always fitted by Nelder-Mead.
:fit_device: *(str, default: the value of ``device``)* Device of the hyperparameter fit, ``"cpu"`` or ``"gpu"``. Several searches sharing a consumer GPU (FP64 at 1/64 of the FP32 rate) are fastest with ``fit_device="cpu"``, or half of them with each device under NVIDIA MPS; a single search is fastest with the fit on the GPU. Several processes on one machine: limit NumPy's BLAS to one thread per process (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1``); its thread pools otherwise oversubscribe the cores (14 processes: 35 s instead of 50 s for 112 light curves).
:fit_bin_width: *(float, default: 10 min)* The fit uses data binned to this width (days) if the cadence is shorter than half of it. ``None``: never bin. The search always uses the unbinned data.

Search grid (as in TLS):

:period_min, period_max, oversampling_factor, duration_grid_step: Period and duration grids, see :doc:`Python interface`.
:n_transits_min: *(int, default: 2)* Minimum number of transits with data.
:grid_transits_min: *(int)* The longest trial period is the time span divided by this (default: ``n_transits_min``). Two transits fit into the time span at all periods below it, so ``n_transits_min=2, grid_transits_min=1`` searches all periods up to the time span with at least two transits.
:R_star_min, R_star_max, M_star_min, M_star_max: *(float, default: TLS defaults)* Stellar limits of the duration grid.
:duration_min_scale: *(float, default: 1)* Factor on the shortest trial duration of each period. The limit belongs to a central transit; :math:`\sqrt{1-b^2}` extends it to impact parameter :math:`b` (0.44 for :math:`b = 0.9`).
:t0_step, t0_step_duration: Step of the mid-transit time grid: by default a quarter of the cadence, coarsened to ``t0_step_duration`` (0.1) times the duration for long durations.

Light curves made of segments (e.g., the quarters of a space mission, with offsets or drifts between them):

:segment_gap: *(float)* Gaps longer than this (days) separate segments. Each segment is normalised by its median and divided by a polynomial; the baseline is fitted once, on all segments, and in the search each segment has its own filter and free mean. Default ``None``: one filter over the whole light curve.
:segment_degree: *(int, default: 4)* Degree of the polynomial per segment, fitted with sigma clipping (``None``: no polynomial).

Computation:

:device: *(str, default: "cpu")* ``"gpu"``: an NVIDIA GPU (see :doc:`Installation`). Search and fold in single precision, the likelihood and the Kalman passes in double precision (the gradient of the fit in single precision). The results equal the CPU results up to rounding.
:fold: *(str, default: "blocked")* The period fold: ``"blocked"`` (every epoch of every period), or the approximate ``"ffa"`` (fast folding, faster on long light curves on the CPU), with options ``-screen`` and ``-coarse``.
:use_threads: *(int)* Number of CPU threads (default: numba's setting).
:trend: *(bool, default: True)* Compute the trend of the result (the posterior mean of the GP outside the transits of the best signal).
:bayes_factor: *(bool, default: False)* Also compute log Bayes factors per period (slower; CPU only).
:timings: *(dict)* If given, filled with the wall time of each stage.


Results
------------------------

``gp.search`` returns a dict whose items are also attributes:

:period, T0, duration: Best signal (days); ``T0`` is the first transit time on the grid.
:depth, depth_err, snr: Depth at the bottom of the template (relative flux), its uncertainty, and their ratio.
:transit_times: Mid-transit times of the best signal.
:periods, power, SDE, SDE_raw: Trial periods, SDE spectrum and its maximum, as in TLS.
:delta_chi2, chi2, chi2_0: Best :math:`\chi^2` improvement per period, :math:`\chi^2 = \chi^2_0 - \Delta\chi^2`, and :math:`\chi^2_0` without a transit.
:durations: Trial durations.
:baseline, baseline_log_likelihood: The fitted (or given) baseline and its log likelihood.
:trend: Posterior mean of the baseline at all times (``None`` with ``trend=False``).
:time, flux, flux_err, use: The data as searched (cleaned, sorted; with ``segment_gap``: normalised per segment and divided by its polynomial), and the points used (``False``: flares).
:log_bf, log_bf_global: Log Bayes factors (with ``bayes_factor=True``).


GPU
------------------------

With ``device="gpu"``, almost all of the work runs on the GPU: on an RTX 5060 (8 GB), a two-year PLATO light curve of :math:`2.5\times10^6` points takes about 9 s including the fit, and a light curve of :math:`10^5` points a few seconds. Further CPU cores hardly help. The TLS search of a detrended light curve can use the GPU through ``power(backend="events-gpu")``.
