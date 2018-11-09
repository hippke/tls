import numpy
import batman
from TransitLeastSquares import rp_rs_from_depth


a = 0.4
b = 0.5
ma = batman.TransitParams()
ma.t0 = 0 # time of inferior conjunction; first transit is X days after start
ma.per = 365.25  # orbital period
ma.rp = 63710 / 696342  # 6371 planet radius (in units of stellar radii)
ma.a = 217  # semi-major axis (in units of stellar radii)
ma.inc = 90  # orbital inclination (in degrees)
ma.ecc = 0  # eccentricity
ma.w = 90  # longitude of periastron (in degrees)
ma.u = [a, b]  # limb darkening coefficients
ma.limb_dark = "quadratic"  # limb darkening model
t = numpy.linspace(-1, 1, 10000)
m = batman.TransitModel(ma, t)  # initializes model
f = m.light_curve(ma)  # calculates light curve

rprs = rp_rs_from_depth(depth=min(f), a=a, b=b)
print(rprs, ma.rp)
numpy.testing.assert_almost_equal(rprs, ma.rp, decimal=4)
print('Test passed')