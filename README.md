![Logo](https://raw.githubusercontent.com/hippke/tls/master/docs/source/logo.png)
### An optimized transit-fitting algorithm to search for periodic transits of small planets
[![Image](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hippke/tls/blob/master/LICENSE)
[![Image](https://img.shields.io/badge/Python-2.7%20%26%203.5%2B-blue.svg)](https://pypi.org/project/transitleastsquares/)
[![Image](https://img.shields.io/badge/pip%20install-transitleastsquares-blue.svg)](https://pypi.org/project/transitleastsquares/)
[![Image](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://transitleastsquares.readthedocs.io/en/latest/index.html)
[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/hippke/tls/tree/master/tutorials)
[![Image](https://img.shields.io/badge/arXiv-1901.02015-blue.svg)](https://arxiv.org/abs/1901.02015)


## Motivation
We present a new method to detect planetary transits from time-series photometry, the *Transit Least Squares* (TLS) algorithm. While the commonly used Box Least Squares [(BLS, Kovács et al. 2002)](http://adsabs.harvard.edu/abs/2002A%26A...391..369K) algorithm searches for rectangular signals in stellar light curves, *TLS* searches for transit-like features with stellar limb-darkening and including the effects of planetary ingress and egress. Moreover, *TLS* analyses the entire, unbinned data of the phase-folded light curve. These improvements yield a ~10 % higher detection efficiency (and similar false alarm rates) compared to BLS. The higher detection efficiency of our freely available Python implementation comes at the cost of higher computational load, which we partly compensate by applying an optimized period sampling and transit duration sampling, constrained to the physically plausible range. A typical Kepler K2 light curve, worth of 90 d of observations at a cadence of 30 min, can be searched with *TLS* in 10 seconds real time on a standard laptop computer, just as with BLS.

![image](https://raw.githubusercontent.com/hippke/tls/master/docs/source/frontpage_rescaled.png)

## Installation

TLS can be installed conveniently using: `pip install transitleastsquares`

If you have multiple versions of Python and pip on your machine, try: `pip3 install transitleastsquares`

The latest version can be pulled from github::
```
git clone https://github.com/hippke/tls.git
cd tls
python setup.py install
```

If the command `python` does not point to Python 3 on your machine, you can try to replace the last line with `python3 setup.py install`. If you don't have `git` on your machine, you can find installation instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). TLS also runs on Python 2, but without multi-threading.


Dependencies:
Python 3,
[NumPy](http://www.numpy.org/),
[numba](http://numba.pydata.org/),
[batman-package](https://www.cfa.harvard.edu/~lkreidberg/batman/),
[tqdm](https://github.com/tqdm/tqdm),
optional:
[argparse](https://docs.python.org/3/library/argparse.html) (for the command line version),
[astroquery](https://astroquery.readthedocs.io/en/latest/) (for LD and stellar density priors from Kepler K1, K2, and TESS).

If you have trouble installing, please [open an issue](https://github.com/hippke/tls/issues).


## Getting started
Here is a short animation of a real search for planets in Kepler K2 data. For more examples, have a look at the [tutorials](https://github.com/hippke/tls/tree/master/tutorials) and the [documentation](https://transitleastsquares.readthedocs.io/en/latest/index.html).

![image](https://raw.githubusercontent.com/hippke/tls/master/docs/source/animation.gif)

## Attribution
Please cite [Hippke & Heller (2019, A&A 623, A39)](https://ui.adsabs.harvard.edu/#abs/2019A&A...623A..39H/abstract) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{2019A&A...623A..39H,
       author = {{Hippke}, Michael and {Heller}, Ren{\'e}},
        title = "{Optimized transit detection algorithm to search for periodic transits of small planets}",
      journal = {\aap},
         year = "2019",
        month = "Mar",
       volume = {623},
          eid = {A39},
        pages = {A39},
          doi = {10.1051/0004-6361/201834672},
archivePrefix = {arXiv},
       eprint = {1901.02015},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2019A&A...623A..39H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contributing Code, Bugfixes, or Feedback
We welcome and encourage contributions. If you have any trouble, [open an issue](https://github.com/hippke/tls/issues).

Copyright 2019 Michael Hippke & René Heller.
