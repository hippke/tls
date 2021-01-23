Python Interface
================

This describes the Python interface to TLS.


Define data for a search
------------------------

.. class:: transitleastsquares.model(t, y, dy)

:t: *(array)* Time series of the data (**in units of days**)
:y: *(array)* Flux series of the data, so that ``1`` is nominal flux (out of transit) and ``0`` is darkness. A transit may be represented by a flux of e.g., ``0.99``
:dy: *(array, optional)* Measurement errors of the data

.. note::

   TLS works best with a constant cadence. Variations in the cadence generally have a negligible impact on detection efficiency, but may result in incorrect transit duration estimates. Small variations, e.g. from the barycentering of the Kepler satellite, can usually be neglected.

.. note::
   Gaps in the data during a transit may decrease detection efficiency. The effect becomes negligible for a large number of transits (e.g., 20), but may be relevant in case of a few (e.g., 3) transits. Then, sorting the data points in phase space may result in an asymmetric transit shape, reducing detection efficiency when using normal (symmetric) transit shape templates.

.. note::
   The time series must be **in units of days**. This is not a chicanery, but a necessity based on the physical model which is used to reduce the parameter space. The unit *day* is a logic choice as orbital periods are typically given in *days*. The Kepler mission also used this unit.



Define parameters and run search
--------------------------------

.. class:: transitleastsquares.power(parameters)

Parameters used for the period search grid and the transit duration search grid. All parameters are optional.

:R_star: *(float, default: 1.0)* Stellar radius (in units of solar radii)
:R_star_min: *(float, default: 0.13)* Minimum stellar radius to be considered (in units of solar radii)
:R_star_max: *(float, default: 3.5)* Maximum stellar radius to be considered (in units of solar radii)
:M_star: *(float, default: 1.0)* Stellar mass (in units of solar masses).
:M_star_min: *(float, default: 0.1)* Minimum stellar mass to be considered (in units of solar masses)
:M_star_max:  *(float, default: 1.0)* Maximum stellar mass to be considered (in units of solar masses)

:period_min:  *(float)* Minimum trial period (in units of days). If none is given, the limit is derived from the Roche limit
:period_max: *(float)* Maximum trial period (in units of days). Default: Half the duration of the time series
:n_transits_min: *(int, default: 2)* Minimum number of transits required. Overrules ``period_max=time_span/n_transits_min``

.. note::

   A larger range of stellar radius and mass allows for a wider variety of transits to be found at the expense of computational effort



Physical parameters to create a
`Mandel & Agol (2002) <https://ui.adsabs.harvard.edu/#abs/2002ApJ...580L.171M/abstract>`_ transit model using a subset of the
`batman module <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_  and syntax (`Kreidberg 2015 <https://ui.adsabs.harvard.edu/#abs/2015PASP..127.1161K/abstract>`_). Available defaults are described below.

:per: *(float)* Orbital period (in units of days). Default: X.
:rp: *(float)* Planet radius (in units of stellar radii). Default: X.
:a: *(float)* Semi-major axis (in units of stellar radii). Default: X.
:inc: *(float)* Orbital inclination (in degrees). Default: 90.
:b: *(float)* Orbital impact parameter as the sky-projected distance between the centre of the stellar disc and the centre of the planetary disc at conjunction. If set, overrules ``inc=degrees(arccos(b/a)``. Default: 0.
:ecc: *(float)* Orbital eccentricity. Default: 0.
:w: *(float)* Argument of periapse (in degrees). Default: 90.
:u: *(array)* List of limb darkening coefficients. Default: [0.4804, 0.1867] (a G2V star in the Kepler bandpass).
:limb_dark: *(str)* Limb darkening model (choice of ``nonlinear``, ``quadratic``, ``exponential``, ``logarithmic``, ``squareroot``, ``linear``, ``uniform``, or ``power2``. Default: ``quadratic``.

Available defaults for the physical parameters of the transit model. When set, the individual parameters are overruled. A different search can be performed to look for comet-like transits based on the equations from `Kennedy et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.5587K/exportcitation>`_ where the deepest transit depth is, however, obtained from the batman transit model. There is also an additional option to incorporate a user-defined transit template by injecting custom implentation of `TransitTemplateGenerator <https://github.com/hippke/tls/tree/master/transitleastsquares/template_generator/TransitTemplateGenerator>`_.

:transit_template: *(str)* Choice of ``default``, ``grazing``, ``box``, ```tailed`` and ``custom``.
:transit_template_generator: *(TransitTemplateGenerator)* When transit_template=``custom``, this input parameter needs to be fed into TLS as an object instantiated from a class extending the TLS TransitTemplateGenerator class.


Parameters to balance detection efficiency and computational requirements:

:duration_grid_step: *(float, default: 1.1)* Grid step width between subsequent trial durations, so that :math:`{\rm dur}_{n+1}={\rm dur}_n \times {\rm duration\_grid\_step}`. With the default value of 1.1, each subsequent trial duration is longer by 10%
:transit_depth_min: *(float, default: 10 ppm)* Shallowest transit depth to be fitted. Transit depths down to half the transit_depth_min can be found at reduced sensitivity. A reasonable value should be estimated from the data to balance sensitivity and avoid fitting the noise floor. Overfitting may cause computational requirements larger by a factor of 10. For reference, the shallowest known transit is 11.9 ppm (Kepler-37b, `Barclay et al. 2013 <http://adsabs.harvard.edu/abs/2013Natur.494..452B>`_)
:oversampling_factor: *(int, default: 3)* Oversampling of the period grid to avoid that the true period falls in between trial periods and is missed.
:T0_fit_margin: *(float, default: 0.01)* Acceptable error margin of the mid-transit time T0. Unit: fraction of the transit duration (0.01 is 1%). For small datasets (e.g., Kepler K2; generally: <10k datapoints), this can be set to 0 with minor speed penalty (seconds). Then, every single cadence is sampled. In data with many cadences, however, this can take very long and can have negligible benefits. As an example, consider a Kepler LC light curve of 60000 points, with a maximum fractional transit duration :math:`T_{14}/P=0.12`. The longest phase-folded transit signal to be tested is then 7200 points long. With Kepler noise, shifting this signal point-by-point is overkill. Shifting by 1% of the transit duration would result in shifts of 72 cadences for this specific signal.

.. note::

   Higher ``oversampling_factor`` increases the detection efficiency at the cost of a linear increase in computational effort. Reasonable values may be 2-5 and should be tested empirically for the actual data. An upper limit can be found when the period step is smaller than the cadence, so that the error from shifting the model by one data point in phase dominates over the period trial shift. For a planet with a 365-day period orbiting a solar mass and radius star, this parity is reached for ``oversampling_factor=9`` at 30 min cadence (Kepler LC). Shorter periods have reduced oversampling benefits, as the cadence becomes a larger fraction of the period.
   
   
Parameters to adjust the computational load and the user experience:

:use_threads: *(int)* Number of parallel threads to be used. A processor like the Intel Core i7-8700K has 6 cores and can run 12 threads in parallel using hyperthreading. Setting ``use_threads=12`` will cause a full load. If no parameter is given, TLS determines the number of available threads and uses the maximum available (in this case: 12).
:show_progress_bar: *(bool, default: True)* When set to ``False``, no progress bar (using ``tqdm``) will be shown

.. note::

   Multi-threading (``use_threads>1`) only works with TLS running on Python 3 as of now. On Python 2, TLS should work, but will fall back to single-core.




Return values
------------------------
.. _returnvalues:

The TLS spectra:

:periods: *(array)* The period grid used in the search
:power: *(array)* The power spectrum per period as defined in the TLS paper. We recommend to use this spectrum to assess transit signals. It is the median-smoothed ``power_raw`` spectrum.
:power_raw: *(array)* The raw power spectrum (without median smoothing) as defined in the TLS paper
:SR: *(array)* Signal residue similar to the BLS SR
:chi2: *(array)* Minimum chi-squared (:math:`\chi^2`) per period
:chi2red: *(array)* Minimum chi-squared per degree of freedom (:math:`\chi^2_{\nu}=\chi^2/\nu`) per period, where  :math:`\nu=n-m` with :math:`n` as the number of observations, and :math:`m=4` as the number of fitted parameters (period, T0, transit duration, transit depth).

The TLS statistics:

:SDE: *(float)* Maximum of ``power``
:SDE_raw: *(float)* Maximum of ``power_raw``
:chi2_min: *(float)* Minimum of ``chi2``
:chi2red_min: *(float)*  Minimum of ``chi2red``

Additional transit statistics based on the ``power`` spectrum:

:period: *(float)* Period of the best-fit signal
:period_uncertainty: *(float)* Uncertainty of the best-fit period (half width at half maximum)
:T0: *(float)* Mid-transit time of the first transit within the time series
:duration: *(float)* Best-fit transit duration
:depth: *(float)* Best-fit transit depth (measured at the transit bottom)
:depth_mean: *(tuple of floats)* Transit depth measured as the mean of all intransit points. The second value is the standard deviation of these points multiplied by the square root of the number of intransit points
:depth_mean_even: *(tuple of floats)* Mean depth and uncertainty of even transits (1, 3, ...)
:depth_mean_odd: *(tuple of floats)* Mean depth and uncertainty of odd transits (2, 4, ...)
:rp_rs: *(float)* Radius ratio of planet and star using the analytic equations from `Heller 2019 <https://arxiv.org/abs/1901.01730>`_
:transit_depths: *(array)* Mean depth of each transit
:transit_depths_uncertainties: *(array)* Uncertainty (1-sigma) of the mean depth of each transit
:snr: *(float)* Signal-to-noise ratio. Definition: :math:`{\rm SNR} = \frac{d}{\sigma_o}n^{1/2}` with :math:`d` as the mean transit depth,  :math:`\sigma` as the standard deviation of the out-of-transit points, and :math:`n` as the number of intransit points (`Pont et al. 2006 <https://ui.adsabs.harvard.edu/#abs/2006MNRAS.373..231P/abstract>`_)
:snr_per_transit: *(array)* Signal-to-noise ratio per individual transit
:snr_pink_per_transit: *(array)* Signal-to-pink-noise ratio per individual transit as defined in `Pont et al. (2006) <https://ui.adsabs.harvard.edu/#abs/2006MNRAS.373..231P/abstract>`_
:odd_even_mismatch: *(float)* Significance (in standard deviations) between odd and even transit depths. Example: A value of 5 represents a :math:`5\,\sigma` confidence that the odd and even depths have different depths
:transit_times: *(array)* The mid-transit time for each transit within the time series
:per_transit_count: *(array)* Number of data points during each unique transit
:transit_count: *(int)* The number of transits
:distinct_transit_count: *(int)* The number of transits with intransit data points
:empty_transit_count: *(int)* The number of transits with no intransit data points
:FAP:  *(float)* The false alarm probability for the SDE assuming white noise. Returns NaN for FAP>0.1.
:before_transit_count: *(int)* * Number of data points in transit (phase-folded)
:in_transit_count: *(int)* Number of data points in a bin of length transit duration before transit (phase-folded)
:after_transit_count: *(int)* Number of data points in a bin of length transit duration after transit (phase-folded)


Time series model for visualization purpose:

:model_lightcurve_time: *(array)* Time series spanning ``t``, but without gaps, and oversampled by a factor of 5
:model_lightcurve_model: *(array)* Model flux value of each point in ``model_lightcurve_time``

Phase-folded model for visualization purpose:

:folded_phase: *(array)* Phase of each data point ``y`` when folded to ``period`` so that the transit is at ``folded_phase=0.5``
:folded_y: *(array)* Data flux of each point
:folded_dy: *(array)* Data uncertainty of each point
:model_folded_phase: *(array)* Linear array ``[0..1]`` which can be used to plot the ``model_folded_model``. This is a separate array from ``folded_phase``, because the data may have gaps which would prevent plotting the complete model. This array here is complete.
:model_folded_model: *(array)* Model flux of each point in ``model_folded_phase``


.. note::

   The models are oversampled and calculated for each point in time and phase. This way, the models cover the entire time series (phase space), including gaps. Thus, these curves are not exact representations of the models used during the search. They are intended for visualization purposes.


Period grid
-----------

When searching for sine-like signals, e.g. using Fourier Transforms, it is optimal to uniformly sample the trial frequencies. This was also suggested for BLS `(Kovács et al. 2002) <https://ui.adsabs.harvard.edu/#abs/2002A&A...391..369K/abstract>`_. However, when searching for transit signals, this is not optimal due to the transit duty cycle which changes as a function of the planetary period due to orbital mechanics. The optimal period grid, compared to a linear grid, reduces the workload (at the same detection efficiency) by a factor of a few. The optimal frequency sampling as a function of stellar mass and radius was derived by `Ofir (2014) <https://ui.adsabs.harvard.edu/#abs/2014A&A...561A.138O/abstract>`_ as

.. math:: N_{\rm freq,{ }optimal} = \left( f_{\rm max}^{1/3} - f_{\rm min}^{1/3} + \frac{A}{3} \right) \frac{3}{A}

with

.. math:: A=\frac{(2\pi)^{2/3}}{\pi }\frac{R}{(GM)^{1/3}}\frac{1}{S \times OS}

where :math:`M` and :math:`R` are the stellar mass and radius, :math:`G` is the gravitational constant, :math:`S` is the time span of the dataset and :math:`OS` is the oversampling parameter to ensure that the peak is not missed between frequency samples. The search edges can be found at the Roche limit,

.. math:: f_{\rm max}=\frac{1}{2 \pi} \sqrt{\frac{GM}{(3R)^3}}; f_{\rm min}=2/S

.. function:: period_grid(parameters)

:R_star: Stellar radius (in units of solar radii)
:M_star: Stellar mass (in units of solar masses)
:time_span: Duration of time series (in units of days)
:period_min:  Minimum trial period (in units of days). Optional.
:period_max: Maximum trial period (in units of days). Optional.
:oversampling_factor: Default: 2. Optional.

    Returns: a 1D array of float values representing a grid of trial periods in units of days.

Example usage:

::

    from transitleastsquares import period_grid
    periods = period_grid(R_star=1, M_star=1, time_span=400)

returns a period grid with 32172 values:

::

    [200, 199.889, 199.779, ..., 0.601, 0.601, 0.601]

.. note::
    TLS calls this function automatically to derive its period grid. Calling this function separately can be useful to employ a classical BLS search, e.g., using the astroPy BLS function.


.. note::
    To avoid generating an infinitely large period_grid, parameters are auto-enforced to the ranges ``0.1 < R_star < 10000`` and ``0.01 < M_star < 1000``. Some combinations of mostly implausible values, such as ``R_star=1`` with ``M_star=5`` yield empty period grids. If the grid size is less than 100 values, the function returns the default grid ``R_star=M_star=1``. Very short time series (less than a few days of duration) default to a grid size with a span of 5 days.





Priors for stellar parameters
--------------------------------

This function provides priors for stellar mass, radius, and limb darkening for stars observed during the Kepler K1, K2 and TESS missions. It is planned to extend this function for past and future missions such as CHEOPS and PLATO.

.. function:: catalog_info(EPIC_ID or TIC_ID)

:EPIC_ID: *(int)* The EPIC catalog ID (K2, Ecliptic Plane Input Catalog)
:TIC_ID: *(int)* The TIC catalog ID (TESS Input Catalog)
:KIC_ID: *(int)* The Kepler Input Catalog ID (Kepler K1 Input Catalog)

Returns

:ab: *(tuple of floats)* Quadratic limb darkening parameters a, b
:mass: *(float)* Stellar mass (in units of solar masses)
:mass_min: *(float)* 1-sigma upper confidence interval on stellar mass (in units of solar mass)
:mass_max: *(float)* 1-sigma lower confidence interval on stellar mass (in units of solar mass)
:radius: *(float)* Stellar radius (in units of solar radii)
:radius_min: *(float)* 1-sigma upper confidence interval on stellar radius (in units of solar radii)
:radius_max: *(float)* 1-sigma lower confidence interval on stellar radius (in units of solar radii)

.. note::

   The matching between the stellar parameter table and the limb darkening table is performed by first finding the nearest :math:`T_{\rm eff}`, and subsequently the nearest :math:`{\rm logg}`.

.. note::
    **Data sources:**

    K1 data are pulled from the catalog for Revised Stellar Properties of Kepler Targets (`Mathur et al. 2017 <https://ui.adsabs.harvard.edu/?#abs/2017ApJS..229...30M>`_) with limb darkening coefficients from `Claret et al. (2012, 2013) <https://ui.adsabs.harvard.edu/#abs/2012A%26A...546A..14C/abstract>`_. Data are pulled from Vizier using AstroQuery and matched to limb darkening values saved locally in a CSV file within the TLS package.

    K2 data are collated from the K2 Ecliptic Plane Input Catalog (`Huber et al. 2016 <https://ui.adsabs.harvard.edu/#abs/2016ApJS..224....2H/abstract>`_) with limb darkening coefficients from `Claret et al. (2012, 2013) <https://ui.adsabs.harvard.edu/#abs/2012A%26A...546A..14C/abstract>`_. Data are pulled from Vizier using AstroQuery and matched to limb darkening values saved locally in a CSV file within the TLS package.

    TESS data are collated from the TESS Input Catalog (`TIC, Stassun et al. 2018 <http://adsabs.harvard.edu/abs/2017arXiv170600495S>`_) with limb darkening coefficients from `Claret et al. (2017) <https://ui.adsabs.harvard.edu/?#abs/2017A%26A...600A..30C>`_. TIC data are pulled from `MAST <https://archive.stsci.edu/tess/>`_ and matched to limb darkening values saved locally in a CSV file within the TLS package.


.. warning::

   Upper and lower confidence intervals may be identical. Radius confidence interval may be identical to the radius. Values not available in the catalog are returned as ``None``. When feeding these values to TLS, make sure to validate accordingly.


Example usage:

::

    ab, R_star, R_star_min, R_star_max, M_star, M_star_min, M_star_max = catalog_info(EPIC_ID=211611158)
    print('Quadratic limb darkening a, b', ab[0], ab[1])
    print('Stellar radius', R_star, '+', R_star_max, '-', R_star_min)
    print('Stellar mass', M_star, '+', M_star_max, '-', M_star_min)

produces these results:

::

    Quadratic limb darkening a, b 0.4899 0.1809
    Stellar radius 1.055 + 0.12 - 0.1
    Stellar mass 1.267 + 0.64 - 0.286


.. note::

   Missing catalog entries will be returned as NaN values. These have to be treated on the user side.



Transit mask
--------------------------------

Can be used to plot in-transit points in a different color, or to cleanse the data from a transit signal before a subsequent TLS run to search for further planets.


.. function:: transit_mask(t, period, duration, T0)

:t: *(array)* Time series of the data (in units of days)
:period: *(float)* Transit period e.g. from results: ``period``
:duration: *(float)* Transit duration e.g. from results: ``duration``
:T0: *(float)* Mid-transit of first transit e.g. from results: ``T0``

Returns

:intransit: *(numpy array mask)* A numpy array mask (of True/False values) for each data point in the time series. ``True`` values are in-transit.


Example usage:

::

    intransit = transit_mask(t, period, duration, T0)
    print(intransit)
    >>> [False False False ...]
    plt.scatter(t[in_transit], y[in_transit], color='red')  # in-transit points in red
    plt.scatter(t[~in_transit], y[~in_transit], color='blue')  # other points in blue



Data cleansing
--------------------------------

TLS may not work correctly with corrupt data, such as arrays including values as NaN, None, infinite, or negative. Masked numpy arrays may also be problematic, e.g., when performing a ``transit_mask``. When in doubt, it is recommended to clean the data from masks and non-floating point values. For this, TLS offers a convenience function:

.. function:: cleaned_array(t, y, dy)

:t: *(array)* Time series of the data (in units of days)
:y: *(array)* Flux series of the data
:dy: *(array, optional)* Measurement errors of the data

Returns

Cleaned arrays, where values of type NaN, None, +-inf, and negative have been removed, as well as masks. Removed values make the output arrays shorter.

Example usage:

::

    from transitleastsquares import cleaned_array
    dirty_array = numpy.ones(10, dtype=object)
    time_array = numpy.linspace(1, 10, 10)
    dy_array = numpy.ones(10, dtype=object)
    dirty_array[1] = None
    dirty_array[2] = numpy.inf
    dirty_array[3] = -numpy.inf
    dirty_array[4] = numpy.nan
    dirty_array[5] = -99
    print(time_array)
    print(dirty_array)

    >>> [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
    >>> [1 None inf -inf nan -99 1 1 1 1]

    t, y, dy = cleaned_array(time_array, dirty_array, dy_array)
    print(t)
    print(y)
    >>> [ 1.  7.  8.  9. 10.]
    >>> [1. 1. 1. 1. 1.]





Data resampling (binning)
--------------------------------

TLS run times are strongly dependent on the amount of data. Very roughly, an increase in the data volume by one order of magnitude results in a run time increase of two orders of magnitude (see paper Figure 9).

For a first quick look, or for short cadence data, it may be adequate to down-sample (bin) the data. In general, binning is adequate if there are many data points between two phase grid points at the critical phase sampling.

To bin the data, TLS offers a convenience function:

.. function:: resample(t, y, dy, factor)

:t: *(array)* Time series of the data (in units of days)
:y: *(array)* Flux series of the data
:dy: *(array, optional)* Measurement errors of the data
:factor: *(float, optional, default: 2.0)* Binning factor

Returns

Resampled arrays of length ``len(t)*int(1/factor)``, where the flux (and optionally, dy) values are binned by `linear interpolation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.


Example usage:

::

    from transitleastsquares import resample
    time_new, flux_new = resample(time, flux, factor=3.0)

.. note::

   Values of type (NaN, None, +-inf, negative, or empty) lead to undefined behavior. It is recommended to first use ``cleaned_array`` if needed.
