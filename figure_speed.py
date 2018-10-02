import numpy
import scipy
from scipy.signal import medfilt
import matplotlib.pyplot as plt

points = [1000, 4000, 10000, 65000, 1000000, 1950000, 10000000]
data_bls = [0.1, 1, 2, 3, 4, 5, 7]

#plt.rc('font',  family='serif', serif='Computer Modern Roman')
#plt.rc('text', usetex=True)
plt.figure(figsize=(3.75, 3.75 / 1.5))
ax = plt.gca()
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')

plt.plot(points, data_bls, color='blue', linewidth=1)

plt.ylabel(r'Runtime (s)')
plt.xlabel('Data volume (cadences)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**3, 10**7)
plt.ylim(1, 10**4)
plt.plot((4000, 4000), (1, 10**4), linewidth=0.5, color='black', linestyle='dashed')
plt.plot((65000, 65000), (1, 10**4), linewidth=0.5, color='black', linestyle='dashed')
plt.plot((1950000, 1950000), (1, 10**4), linewidth=0.5, color='black', linestyle='dashed')
ax.text(4100, 2.5, 'K2', ha='center', backgroundcolor='white')
ax.text(65000, 2.5, 'K1 LC', ha='center', backgroundcolor='white')
ax.text(1950000, 2.5, 'K1 SC', ha='center', backgroundcolor='white')
plt.text(10000, 100, 'BLS')
plt.savefig('figure_speed.pdf', bbox_inches='tight')

# 4.25 yrs of Kepler K1 data at 1min (SC) and 30min (LC) cadence. K2 90 days at 30min.
# CPU: Intel Core-i7 7700K, 4 cores, 8 threads, 4.5 GHz
# GPU: nVidia GTX1060
