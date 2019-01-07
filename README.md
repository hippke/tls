![Logo](https://raw.githubusercontent.com/hippke/tls/blob/master/docs/source/logo.png)
### An optimized transit-fitting algorithm to search for periodic transits of small planets
[![Image](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hippke/tls/blob/master/LICENSE)
[![Image](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)](https://pypi.org/project/transitleastsquares/)
[![Image](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://transitleastsquares.readthedocs.io/en/latest/index.html)
[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/hippke/tls/tree/master/tutorials)
Add badge: ADS, arxiv, DOI


## Motivation
We present a new method to detect planetary transits from time-series photometry, the *Transit Least Squares* (TLS) algorithm. While the commonly used Box Least Squares [(BLS, Kovács et al. 2002)](http://adsabs.harvard.edu/abs/2002A%26A...391..369K) algorithm searches for rectangular signals in stellar light curves, *TLS* searches for transit-like features with stellar limb-darkening and including the effects of planetary ingress and egress. Moreover, *TLS* analyses the entire, unbinned data of the phase-folded light curve. These improvements yield a ~10 % higher detection efficiency (and similar false alarm rates) compared to BLS. The higher detection efficiency of our freely available Python implementation comes at the cost of higher computational load, which we partly compensate by applying an optimized period sampling and transit duration sampling, constrained to the physically plausible range. A typical Kepler K2 light curve, worth of 90 d of observations at a cadence of 30 min, can be searched with *TLS* in 10 seconds real time on a standard laptop computer, just as with BLS.

![image](https://raw.githubusercontent.com/hippke/tls/blob/master/docs/source/frontpage_rescaled.png)

## Installation
The stable version can be installed via pip: `pip install tls-package`

Dependencies:
Python 3, 
[SciPy](https://www.scipy.org/),
[NumPy](http://www.numpy.org/),
[numba](http://numba.pydata.org/),
[batman](https://www.cfa.harvard.edu/~lkreidberg/batman/),
[tqdm](https://github.com/tqdm/tqdm),
[argparse](https://docs.python.org/3/library/argparse.html)

## Getting started
Here is a short animation of a real search for planets in Kepler K2 data. For more examples, have a look at the [tutorials](https://github.com/hippke/tls/tree/master/tutorials) and the [documentation](https://transitleastsquares.readthedocs.io/en/latest/index.html).

![image](https://raw.githubusercontent.com/hippke/tls/blob/master/docs/source/animation.gif)

## Attribution
Please cite [Hippke & Heller (2019)](http://www.) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@article{abc,
   author = {},
    title = {},
  journal = {},
     year = 2019,
   volume = ,
    pages = {},
   eprint = {},
      doi = {}
}
```

## Contributing Code, Bugfixes, or Feedback
We welcome and encourage contributions. If you have any trouble, [open an issue](https://github.com/hippke/tls/issues).

Copyright 2019 Michael Hippke & René Heller.
