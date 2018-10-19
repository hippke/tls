Interface
=========

This describes the interface to TLS.


Define data for a search
------------------------

.. class:: TransitLeastSquares.model(time, data, errors)

:t: *(array)* Time series of the data (in units of days)
:y: *(array)* Flux series of the data, so that ``1`` is nominal flux (out of transit) and ``0`` is darkness. A transit may be represented by a flux of e.g., ``0.99``
:dy: *(array, optional)* Measurement errors of the data


Define parameters and run search
--------------------------------

.. class:: TransitLeastSquares.power(parameters)

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
:u: *(array)* List of limb darkening coefficients. Default: [X, Y].
:limb_dark: *(str)* Limb darkening model (choice of ``nonlinear``, ``quadratic``, ``exponential``, ``logarithmic``, ``squareroot``, ``linear``, ``uniform``, or ``power2``. Default: ``quadratic``.

Available defaults for the physical parameters of the transit model. When set, the individual parameters are overruled.

:transit_template: *(str)* Choice of ``Earth``, ``Super-Earth``, ``blubb``


Parameters to balance detection efficiency and computational requirements:

:duration_grid_step: *(float, default: 1.1)* Grid step width between subsequent trial durations, so that :math:`{\rm dur}_{n+1}={\rm dur}_n \times {\rm duration\_grid\_step}`. With the default value of 1.1, each subsequent trial duration is longer by 10%
:transit_depth_min: *(float, default: 10 ppm)* Shallowest transit depth to be fitted. Transit depths down to half the transit_depth_min can be found at reduced sensitivity. A reasonable value should be estimated from the data to balance sensitivity and avoid fitting the noise floor. Overfitting may cause computational requirements larger by a factor of 10. For reference, the shallowest known transit is 11.9 ppm (Kepler-37b, `Barclay et al. 2013 <http://adsabs.harvard.edu/abs/2013Natur.494..452B>`_)
:oversampling_factor: *(int, default: 3)* Oversampling of the period grid to avoid that the true period falls in between trial periods and is missed.

.. note::

   Higher ``oversampling_factor`` increases the detection efficiency at the cost of a linear increase in computational effort. Reasonable values may be 2-5 and should be tested empirically for the actual data. An upper limit can be found when the period step is smaller than the cadence, so that the error from shifting the model by one data point in phase dominates over the period trial shift. For a planet with a 365-day period orbiting a solar mass and radius star, this parity is reached for ``oversampling_factor=9`` at 30 min cadence (Kepler LC). Shorter periods have reduced oversampling benefits, as the cadence becomes a larger fraction of the period.

**Return values**

The TLS spectra:

:periods: *(array)* The period grid used in the search
:power: *(array)* The power spectrum per period as defined in the TLS paper. We recommend to use this spectrum to assess transit signals.
:power_raw: *(array)* The raw power spectrum (without median smoothing) as defined in the TLS paper
:SR: *(array)* Signal residue similar to the BLS SR
:chi2: *(array)* Minimum chi-squared (:math:`\chi^2`) per period 
:chi2red: *(array)* Minimum chi-squared per degree of freedom (:math:`\chi^2_{\nu}=\chi^2/\nu`) per period, where  :math:`\nu=n-m` with :math:`n` as the number of observations, and :math:`m=4` as the number of fitted parameters (period, T0, transit duration, transit depth).

The TLS statistics:

:SDE: *(float)* Maximum of ``power``
:SDE_raw: *(float)* Maximum of ``power_raw``
:chi2_min: *(float)* Minimum of ``chi2``
:chi2red_min: *(float)*  Minimum of ``chi2red``

Additional transit statistics:

:period: *(float)* Period of the best-fit signal
:T0: *(float)* Mid-transit time of the first transit within the time series
:duration: *(float)* Best-fit transit duration
:depth: *(float)* Best-fit transit depth (measured at the transit bottom)
:depth_mean: *(tuple of floats)* Transit depth measured as the mean of all intransit points. The second value is the standard deviation of these points multiplied by the square root of the number of intransit points
:depth_mean_odd: *(float)* Mean depth of odd transits (1, 3, ...)
:depth_mean_odd: *(float)* Mean depth of odd transits (2, 4, ...)
:snr: *(float)* Signal-to-noise ratio. Definition: :math:`{\rm SNR} = \frac{d/\sigma} n^{1/2}` with :math:`d` as the mean transit depth,  :math:`\sigma` as the standard deviation of the out-of-transit points, and :math:`n` as the number of intransit points (`Pont et al. 2006 <https://ui.adsabs.harvard.edu/#abs/2006MNRAS.373..231P/abstract>`_)
:snr_per_transit: *(array)* Signal-to-noise ratio per individual transit
:snr_pink_per_transit: *(array)* Signal-to-pink-noise ratio per individual transit as defined in `Pont et al. (2006) <https://ui.adsabs.harvard.edu/#abs/2006MNRAS.373..231P/abstract>`_
:odd_even_mismatch: *(float)* Significance (in standard deviations) between odd and even transit depths. Example: A value of 5 represents a :math:`5\,\sigma` confidence that the odd and even depths are not equal
:empty_transit_count: *(int)* The number of transits with no intransit data points 
:per_transit_count: *(array)* Number of data points during each unique transit
:transit_times: *(array)* The mid-transit time for each transit within the time series

Models for visualization purpose:

:model: *(array)* Model flux at each time value
:model_phase: *(array)* Phase of the phase-folded model
:model_folded: *(array)* Model flux at each phase
:model_data: *(array)* Data flux at each phase

.. note::

   The models are not exact representations of the models used during the search. They should only be used for rough validation purposes. It is planned to improve the visualiziation in a future release.


Period grid
-----------

When searching for sine-like signals, e.g. using Fourier Transforms, it is optimal to uniformly sample the trial frequencies. This was also suggested for BLS `(Kovács et al. 2002) <https://ui.adsabs.harvard.edu/#abs/2002A&A...391..369K/abstract>`_. However, when searching for transit signals, this is not optimal due to the transit duty cycle which changes as a function of the planetary period due to orbital mechanics. The optimal period grid, compared to a linear grid, reduces the workload (at the same detection efficiency) by a factor of a few. The optimal frequency sampling as a function of stellar mass and radius was derived by `Ofir (2014) <https://ui.adsabs.harvard.edu/#abs/2014A&A...561A.138O/abstract>`_ as

.. math:: N_{\rm freq,{ }optimal} = \left( f_{\rm max}^{1/3} - f_{\rm min}^{1/3} + \frac{A}{3} \right) \frac{3}{A}

with

.. math:: A=\frac{(2\pi)^{2/3}}{\pi }\frac{R}{(GM)^{1/3}}\frac{1}{S \times OS}

where :math:`M` and :math:`R` are the stellar mass and radius, :math:`G` is the gravitational constant, :math:`S` is the time span of the dataset and :math:`OS` is the oversampling parameter to ensure that the peak is not missed between frequency samples. The search edges can be found at the Roche limit, 

.. math:: f_{\rm max}=\frac{1}{2 \pi} \sqrt{\frac{GM}{(3R)^3}}; f_{\rm min}=2/S

.. function:: autoperiod(parameters)
:R_star: Stellar radius (in units of solar radii)
:M_star: Stellar mass (in units of solar masses) 
:time_span: Duration of time series (in units of days)
:period_min:  Minimum trial period (in units of days). Optional.
:period_max: Maximum trial period (in units of days). Optional.
:oversampling_factor: Default: 2. Optional.

    Returns: a 1D array of float values representing a grid of trial periods in units of days.

Example usage:

::

    from TransitLeastSquares import autoperiod
    periods = autoperiod(R_star=1, M_star=1, time_span=400)

returns a period grid with 32172 values:

::

    [200, 199.889, 199.779, ..., 0.601, 0.601, 0.601]


EPIC catalog info
-----------------

A convenience function to pull estimates for stellar mass, radius, and limb darkening for stars observed during the Kepler K2 mission. It is planned to extend this function with catalogs for Kepler, TESS, Gaia, CHEOPS, PLATO and others.

Data are collated from the K2 Ecliptic Plane Input Catalog (`Huber et al. 2016 <https://ui.adsabs.harvard.edu/#abs/2016ApJS..224....2H/abstract>`_) with limb darkening coefficients from `Claret et al. (2012, 2013) <https://ui.adsabs.harvard.edu/#abs/2012A%26A...546A..14C/abstract>`_


.. function:: catalog_info(EPIC_id)

:EPIC_id: *(int)* The EPIC catalog ID

Returns

:u: *(float)* Linear limb darkening parameter u
:ab: *(tuple of floats)* Quadratic limb darkening parameters a, b
:mass: *(float)* Stellar mass (in units of solar masses)
:mass_min: *(float)* 1-sigma upper confidence intervall on stellar mass (in units of solar mass)
:mass_max: *(float)* 1-sigma lower confidence intervall on stellar mass (in units of solar mass)
:radius: *(float)* Stellar radius (in units of solar radii)
:radius_min: *(float)* 1-sigma upper confidence intervall on stellar radius (in units of solar radii)
:radius_max: *(float)* 1-sigma lower confidence intervall on stellar radius (in units of solar radii)
