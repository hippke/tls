import numpy
import matplotlib.pyplot as plt
from math import pi 


def T14max(R_s, M_s, P):
    """Input:  Stellar radius and mass; planetary period
               Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    # astrophysical constants
    G = 6.673e-11  # [m^3 / kg / s^2]
    R_sun = 695508000  # [m]
    M_sun = 1.989*10**30  # [kg]

    P = P * 24 * 3600
    R_s = R_sun * R_s
    M_s = M_sun * M_s

    T14max = R_s * ((4*P) / (pi * G * M_s))**(1/3)

    return T14max / P


# Load data
data = numpy.genfromtxt(
    'data_per_t14.csv',
    dtype='f8, f8',
    names = ['per', 't14'])

# Make figure
#plt.rc('font',  family='serif', serif='Computer Modern Roman')
#plt.rc('text', usetex=True)
size = 3.75
aspect_ratio = 1.5
plt.figure(figsize=(size, size / aspect_ratio))
ax = plt.gca()
ax.set_xlabel(r'Period (days)')
ax.set_ylabel(r'Transit duration ($T_{\rm 14}/P$)')
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2, 1000)
plt.ylim(0.0003, 0.2)

# Plot data
ax.scatter(data['per'], data['t14'], s=10, alpha=0.5, linewidths=0.5, edgecolor='black')

# Bottom border
periods = numpy.geomspace(0.1, 1000, 2)
ax.plot(periods, T14max(R_s=0.184, M_s=1, P=periods), linewidth=1, color='red')
ax.plot(periods, T14max(R_s=0.13, M_s=0.1, P=periods), linewidth=1, linestyle='dashed', color='red')

# M8, G2, A5
periods = numpy.geomspace(0.8, 1000, 2)
ax.plot(periods, T14max(R_s=1.7, M_s=2.1, P=periods), linewidth=1, linestyle='dashed', color='red')
periods = numpy.geomspace(0.5, 1000, 2)
ax.plot(periods, T14max(R_s=1, M_s=1, P=periods), linewidth=1, linestyle='dashed', color='red')

#ax.plot(periods, T14max(R_s=6.13, M_s=1, P=periods), linewidth=1, color='red')
# Upper limits
periods = numpy.geomspace(3.3, 1000, 2)
ax.plot(periods, T14max(R_s=3.5, M_s=1, P=periods), linewidth=1, color='red')
ax.plot((0.01, 3.3), (0.12, 0.12), linewidth=1, color='red')

ax.text(200, 0.00045, 'M8')
ax.text(500, 0.0005, 'G2')
ax.text(500, 0.002, 'A5')
plt.savefig('figure_per_t14.pdf', bbox_inches='tight')
