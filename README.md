![Logo](https://github.com/hippke/tls/blob/master/logo.png)
### An optimized transit-fitting algorithm to search for periodic transits of small planets
[![Image](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hippke/tls/blob/master/LICENSE "MIT license")
[![Image](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)](https://pypi.org/project/tls-package/ "PyPI")
Add badge: ADS, arxiv, DOI, ASCL

Work in progress. Do not use :-)



## Motivation
We present a new method to detect planetary transits from time-series photometry, the *Transit Least Squares* (TLS) algorithm. While the commonly used Box Least Squares [(BLS, Kovács et al. 2002)](http://adsabs.harvard.edu/abs/2002A%26A...391..369K) algorithm searches for rectangular signals in stellar light curves, *TLS* searches for transit-like features with stellar limb-darkening and including the effects of planetary ingress and egress. Moreover, *TLS* analyses the entire, unbinned data of the phase-folded light curve and it calculates the model transit light curve from an oversampled light curve to account for the temporal smearing effects of finite exposures during observations. These improvements yield a 5-10 % higher detection efficiency (and similar false alarm rates) compared to BLS. The higher detection efficiency of our freely available Python implementation comes at the cost of higher computational load, which we partly compensate by applying an optimized period sampling and transit duration sampling, constraint to the physically plausible range. A typical K2 light curve, worth of 90 d of observations at a cadence of 30 min, can be searched with *TLS* in 3 min real time on a standard laptop computer, compared to 25 s with BLS.

![image](https://github.com/hippke/tls/blob/master/frontpage_rescaled.png)

## Tutorial
Open the [iPython tutorial](https://github.com/hippke/tls/blob/master/tls_tutorial.ipynb) for a quick introduction.

## Documentation
Open the [complete documentation](https://www).

## Installation
`pip install tls-package`

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
