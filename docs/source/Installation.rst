Installation
=====================================

TLS can be installed conveniently using pip::

    pip install transitleastsquares

The latest version can be pulled from github::

    git clone https://github.com/hippke/tls.git
    cd tls
    python setup.py install



Compatibility
------------------------

TLS has been `tested to work <https://travis-ci.com/hippke/tls>`_ with Python 3.5, 3.6, 3.7. It fails on Python 2.7 due to the lack of proper multi-processing support (in particular, `multiple arguments <https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments>`_ are difficult to handle).