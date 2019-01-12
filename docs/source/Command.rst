Command line interface
=========================

This describes the command line interface to TLS.

Usage
------------------------

Syntax:
::

    transitleastsquares.py [-h] [-o OUTPUT] [-c CONFIG] lightcurve

Minimum example:
::

    python transitleastsquares.py test_data.csv


Maximum example:
::

    python transitleastsquares.py test_data.csv --config=tls_config.cfg --output=results.csv


.. note::

   In the current TLS version, custom transit shapes can not be defined with the command line interface. Use the Python interface instead.


Config file
------------------------

Syntax:

::

    [Grid]
    R_star = 1
    R_star_min = 0.13
    R_star_max = 3.5
    M_star = 1
    M_star_min = 0.1
    M_star_max = 1
    period_min = 0
    period_max = 1e10
    n_transits_min = 2

    [Template]
    transit_template = default

    [Speed]
    duration_grid_step = 1.1
    transit_depth_min = 10e-6
    oversampling_factor = 3
    T0_fit_margin = 0.01


Output
------------------------

The output file contains the :ref:`return values <returnvalues>`. One item per line: First the dictionary key, a space as delimiter, then the value(s). Arrays are in the format ``[0, 1, ..., 5]``.

