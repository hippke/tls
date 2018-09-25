import batman
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.colors import LinearSegmentedColormap

#plt.rc('font',  family='serif', serif='Computer Modern Roman')
#plt.rc('text', usetex=True)
size = 4.5
aspect_ratio = 1.5
fig, ax = plt.subplots(figsize=(size, size / aspect_ratio))
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
lds = [0.939, 0.751, 0.570, 0.464, 0.386, 0.333, 0.284, 0.172, 0.132]
Teffs = [300, 400, 500, 600, 700, 800, 900, 1500, 2000]

ma.u = [0.50]
m = batman.TransitModel(ma, t)    # initializes model
flux = m.light_curve(ma)          # calculates light curve
reference_ld_curve = (flux-1)*10**6
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(Teffs), vmax=max(Teffs))
cmap = colors.LinearSegmentedColormap.from_list("", 
    ["Violet","Blue","Green", "Yellow", "Orange", "Red", "Red", "Red", \
    "Red", "Red", "Red",  "Red",  "Red",  "Red",  "Red", "Black"])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

# Series of LD curves
i = 0
for ld in lds:
    ma.u = [ld]
    m = batman.TransitModel(ma, t)    #initializes model
    flux = m.light_curve(ma)          #calculates light curve
    scaled_flux = (flux-1)*10**6
    colorVal = scalarMap.to_rgba(Teffs[i])
    plts = plt.plot(t * 24, scaled_flux, color=colorVal, linewidth=2)
    i = i + 1

plt.xlabel("Time from central transit (hrs)")
plt.ylabel("Relative flux (ppm)")
plt.xlim(-7, 7)
plt.ylim(-130, 10)
x = np.linspace(1000, 10, num=100)
y = np.linspace(min(Teffs), max(Teffs), num=100)
scatters = plt.scatter(x, y, c=y, cmap=cmap)
cbar = fig.colorbar(scatters, ax=ax)
cbar.set_label('Wavelength (nm)', rotation=270, labelpad=15)

plt.savefig('fig_earth_lds_long_wavelengths.pdf', bbox_inches='tight')
