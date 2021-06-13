Command line interface
=========================

This describes the command line interface to TLS. After installation, you can call it from the command line.

Usage
------------------------

Syntax:
::

    transitleastsquares [-h] [-o OUTPUT] [-c CONFIG] lightcurve

Minimum example:
::

    transitleastsquares test_data.csv


Maximum example:
::

    transitleastsquares test_data.csv --config=tls_config.cfg --output=results.csv


.. note::

   In the current TLS version, custom transit shapes can not be defined with the command line interface. If you have a use case for more complex searches using the command line interface, please `open an issue on Github <https://github.com/hippke/tls/issues/new/choose>`_ and I will add it to the next version.


Config file
------------------------

Syntax:

::

    [Grid]
    R_star = 1
    R_star_min = 0.8
    R_star_max = 1.2
    M_star = 1
    M_star_min = 0.8
    M_star_max = 1.2
    period_min = 0
    period_max = 1e10
    n_transits_min = 3

    [Template]
    transit_template = default

    [Speed]
    duration_grid_step = 1.1
    transit_depth_min = 10e-6
    oversampling_factor = 2
    T0_fit_margin = 0.01
    use_threads = 4

    [File]
    delimiter = ,


Output
------------------------

After a successful TLS run, 2 files are generated:
:statistics:  ``lightcurve filename + _statistics.csv``
:SDE-ogram: ``lightcurve filename + _power.csv``
