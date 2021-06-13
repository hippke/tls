Changelog
=========

This describes changes to TLS.

The versioning scheme is: major.minor.revision

:major: Will be increased when the API (the interface) changes in an incompatible way. Will be documented in this changelog.
:minor: Will be increased when adding functionality in a backwards-compatible manner. Will be documented in this changelog.
:revision: Will be increased for backwards-compatible bug fixes and very minor added functionality. Will not always be documented in this changelog.


Version 1.0.23 (12 March 2019)
------------------------------------

:Fixed: A bug in the post-detection statistics which caused a delay in large dataset (e.g., Kepler K1)


Version 1.0.21 (14 February 2019)
------------------------------------

:Fixed: A bug in ``cleaned_array`` which caused an error in case invalid ``dy`` values were supplied. Relevant for TESS FITS files. These should now run including uncertainties.
:Added: New statistics: ``before_transit_count``, ``in_transit_count``, ``after_transit_count`` yield the number of data points in a bin of length transit duration before, in and after the phase-folded transit.
:Added: New parameter ``show_progress_bar`` *(bool, default: True)* When set to ``False``, no progress bar (using ``tqdm``) will be shown
:Added: Python 2 compatibility. For now, it is only single-core.


Version 1.0.20 (11 February 2019)
-----------------------------------

:Fixed: A bug which	erroneously interchanged `power` and `power_raw`


Version 1.0.19 (10 February 2019)
-----------------------------------

:Fixed: A bug in the calculation of statistics which caused TLS to stall in some cases



Version 1.0.17 (07 February 2019)
-----------------------------------

:Fixed: A bug in the calculation of the SNR statistic (post-fit statistics)
:Changed: Major code refactoring
:Added: Extensive test suite
:Added: Warnings for the user, e.g. in case inputs are dubious. Numpy warnings are now a bug (if any left) and no longer hidden.
:Added: Improved command line interface (now has its own command). Added all recent new functionality to the command line interface (all except custom transit shapes)


Version 1.0.16 (29 January 2019)
-----------------------------------

:Fixed: A bug which caused to return an empty SDE-ogram if very small uncertainties ``dy`` were provided.
:Changed: Switched linear interpolation code of model shapes to a numba implementation. It is 2x faster, 20ms --> 10ms which is currently irrelevant if the shape is calculated only once per light curve, but will become relevant when the compensation for morphological light-curve distortions will be implemented. Then, the shapes will be re-calculated many times for a range of periods. Another advantage is that the dependency on scipy can now be removed. Scipy is still required for testing, however.


Version 1.0.15 (27 January 2019)
-----------------------------------

:Changed: If no transits fits were performed during a search, a flat SDE-ogram and SDE=0 are returned, and a warning is raised. Previous behavior was to raise an exception and quit. This can happen if ``transit_depth_min`` is set to a large value (e.g., 1000 ppm) and the light curve is flat (e.g., Kepler-quality with good detrending and no transits), so that the threshold causes no transit fits to be performed.
:Changed: Only useful warnings are printed to the user console. Internal processing issues (e.g., NaN values) are now hidden.
:Changed: Catalog information (e.g., from the Kepler K2 EPIC catalog) which includes missing values now returns ``NaN`` values. Previously, ``--`` was returned. The ``NaN`` values must still be evaluated by the user before feeding them into a TLS model.
:Changed: Catalog information is now entirely pulled using AstroQuery, from Vizier (Kepler K1, K2) and MAST. Dependency to package ``kplr`` has been dropped. This increases reliability as the MAST API was unstable in the past.
:Fixed: A bug in the command-line version was fixed which caused the search to quit under certain circumstances.


Version 1.0.14. (24 January 2019)
----------------------------------

:Added: Automatically run ``cleaned_array`` before performing a search
:Added: New return value: ``results.transit_depths_uncertainties``
:Added: New parameter: ``use_threads``
:Changed: ``period_grid`` limited to physically plausible values to avoid generating empty or extremely large grids
:Removed: ``numpy.set_printoptions(threshold=numpy.nan)`` which fails in numpy 1.16+ (the latest version as of 24 Jan 2019)


Version 1.0 (01 January 2018)
-------------------------------

Initial release.
