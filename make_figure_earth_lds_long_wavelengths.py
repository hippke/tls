import batman
import numpy as np
import matplotlib.pyplot as plt


def box(width, depth):  # width in arbitrary units, depth in ppm
    f = np.zeros(20000)
    f[width:-width] = depth
    #f[-width:20000] = depth
    return f


plt.rc('font',  family='serif', serif='Computer Modern Roman')
plt.rc('text', usetex=True)
size = 4.5
aspect_ratio = 1.5
plt.figure(figsize=(size, size / aspect_ratio))
ax = plt.gca()
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')

ma = batman.TransitParams()
ma.t0 = 0                       #time of inferior conjunction
ma.per = 365.25                    #orbital period
ma.rp =  6371 / 696342             #planet radius (in units of stellar radii)
ma.a = 217               #semi-major axis (in units of stellar radii)
ma.inc = 90          #orbital inclination (in degrees)
ma.ecc = 0                      #eccentricity
ma.w = 90                       #longitude of periastron (in degrees)
ma.u = [0.5]                #limb darkening coefficients
ma.limb_dark = "linear"       #limb darkening model

t = np.linspace(-1, 1, 20000)
#lds = np.linspace(0.33, 0.76, 100)  # all planet host stars

lds = [0.939, 0.751, 0.570, 0.464, 0.386,  0.333, 0.284, 0.172, 0.132]

# 300, 400, 500, 600, 700, 800, 900, 1500, 2000

#lds = np.linspace(0.40, 0.63, 100)  # full range 2300K - 12000K


#lds = np.linspace(0.01, 0.99, 100)  #


ma.u = [0.50]
m = batman.TransitModel(ma, t)    #initializes model
flux = m.light_curve(ma)          #calculates light curve
reference_ld_curve = (flux-1)*10**6
#plt.plot(t * 24, reference_ld_curve, color='black')

# Series of LD curves
for ld in lds:
    ma.u = [ld]
    m = batman.TransitModel(ma, t)    #initializes model
    flux = m.light_curve(ma)          #calculates light curve
    scaled_flux = (flux-1)*10**6
    plt.plot(t * 24, scaled_flux, color='black', linewidth=0.5)
    diff = scaled_flux - reference_ld_curve
    diff = diff**2
    summed_residuals = np.sum(diff)
    print(ld, summed_residuals)


plt.xlabel("Time from central transit (hrs)")
plt.ylabel("Relative flux (ppm)")
plt.xlim(-7, 7)
plt.ylim(-130, 10)
ax.text(-1.5, -5, 'Earth/Sun')
plt.savefig('fig_earth_lds_long_wavelengths.pdf', bbox_inches='tight')
