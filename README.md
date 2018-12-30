![Logo](https://github.com/hippke/tls/blob/master/images/logo.png)
### An optimized transit-fitting algorithm to search for periodic transits of small planets
[![Image](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hippke/tls/blob/master/LICENSE "MIT license")
[![Image](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)](https://pypi.org/project/tls-package/ "PyPI")
Add badge: ADS, arxiv, DOI, ASCL


## Motivation
We present a new method to detect planetary transits from time-series photometry, the *Transit Least Squares* (TLS) algorithm. While the commonly used Box Least Squares [(BLS, Kovács et al. 2002)](http://adsabs.harvard.edu/abs/2002A%26A...391..369K) algorithm searches for rectangular signals in stellar light curves, *TLS* searches for transit-like features with stellar limb-darkening and including the effects of planetary ingress and egress. Moreover, *TLS* analyses the entire, unbinned data of the phase-folded light curve and it calculates the model transit light curve from an oversampled light curve to account for the temporal smearing effects of finite exposures during observations. These improvements yield a 5-10 % higher detection efficiency (and similar false alarm rates) compared to BLS. The higher detection efficiency of our freely available Python implementation comes at the cost of higher computational load, which we partly compensate by applying an optimized period sampling and transit duration sampling, constraint to the physically plausible range. A typical K2 light curve, worth of 90 d of observations at a cadence of 30 min, can be searched with *TLS* in 10 seconds real time on a standard laptop computer, just as with BLS.

![image](https://github.com/hippke/tls/blob/master/images/frontpage_rescaled.png)

## Tutorials
Open the [iPython tutorials](https://github.com/hippke/tls/tree/master/tutorials) for a quick introduction.

## Documentation
Open the [complete documentation](http://jaekle.info/tls/Python%20interface.html).

## Installation
The stable version can be installed via pip: `pip install tls-package`

The current development version can be installed from this repository:
```
git clone https://github.com/hippke/tls.git
cd tls
python3 setup.py install
```

## Attribution
Please cite [Hippke & Heller (2018)](http://www.) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@article{abc,
   author = {},
    title = {},
  journal = {},
     year = ,
   volume = ,
    pages = {},
   eprint = {},
      doi = {}
}
```

## Contributing Code, Bugfixes, or Feedback
We welcome and encourage contributions.

Copyright 2018 Michael Hippke & René Heller.
