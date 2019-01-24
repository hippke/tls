Changelog
=========

This describes changes to TLS.

The versioning scheme is: major.minor.revision

:major: Will be increased when the API (the interface) changes in an incompatible way. Will be documented in this changelog.
:minor: Will be increased when adding functionality in a backwards-compatible manner. Will be documented in this changelog.
:revision: Will be increased for backwards-compatible bug fixes and very minor added functionality. Will not always be documented in this changelog.


Version 1.0.14. (planned)
------------------------------

:Added: Automatically run ``cleaned_array`` before performing a search
:Added: New return value: ``results.transit_depths_uncertainties``
:Added: New parameter: ``use_threads``
:Changed: ``period_grid`` limited to physically plausible values to avoid generating empty or extremely large grids
:Removed: ``numpy.set_printoptions(threshold=numpy.nan)`` which fails in numpy 1.18 and later


Version 1.0 (01 January 2018)
------------------------------

Initial release.
