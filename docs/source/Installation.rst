Installation
=====================================

TLS can be installed conveniently using pip::

    pip install transitleastsquares

If you have multiple versions of Python and pip on your machine, make sure to use pip3. Try::

    pip3 install transitleastsquares


The latest version can be pulled from github::

    git clone https://github.com/hippke/tls.git
    cd tls
    python setup.py install

If the command ``python`` does not point to Python 3 on your machine, you can try to replace the last line with ``python3 setup.py install``. If you don't have git on your machine, you can find installation instructions `here <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.


Compatibility
------------------------

TLS has been `tested to work <https://travis-ci.com/hippke/tls>`_ with Python 3.5, 3.6, 3.7. It fails on Python 2.7 due to the lack of proper multi-processing support (in particular, `multiple arguments <https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments>`_ are difficult to handle).