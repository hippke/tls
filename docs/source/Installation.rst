Installation
=====================================

TLS can be installed conveniently using pip::

    pip install transitleastsquares

If you have multiple versions of Python and pip on your machine, make sure to use pip3. Try::

    pip3 install transitleastsquares


The latest version can be pulled from github::

    git clone https://github.com/hippke/tls.git
    cd tls
    pip install .

For an editable (development) install, use ``pip install -e .`` instead. If you don't have ``git`` on your machine, you can find installation instructions `here <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

Dependencies: Python 3.9+, NumPy (>= 1.22), numba, tqdm. Optional: ``astroquery`` (for limb darkening and stellar density priors from the Kepler K1, K2, and TESS catalogs) and ``batman-package`` (only needed to run the test suite).


Compatibility
------------------------

TLS requires Python 3.9 or later and has been tested with Python 3.9 (NumPy 1.22, numba 0.60) up to Python 3.14 (NumPy 2.5, numba 0.68).

The first search on a machine compiles the numba kernels, which takes a few seconds; the compiled code is cached, so subsequent searches (including those in parallel worker processes) start immediately.
