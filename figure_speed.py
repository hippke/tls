import numpy
import scipy
from scipy.signal import medfilt
import matplotlib.pyplot as plt

points = [1000, 4000, 10000, 65000, 1000000, 1950000, 10000000]
data_bls = [1, 10, 20, 30, 40, 50, 70]

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
plt.plot((4000, 4000), (1, 10**4), linewidth=0.5, color='gray', linestyle='dashed')
plt.plot((18000, 18000), (1, 10**4), linewidth=0.5, color='gray', linestyle='dashed')
plt.plot((65000, 65000), (1, 10**4), linewidth=0.5, color='gray', linestyle='dashed')
plt.plot((216000, 216000), (1, 10**4), linewidth=0.5, color='gray', linestyle='dashed')
plt.plot((1950000, 1950000), (1, 10**4), linewidth=0.5, color='gray', linestyle='dashed')
ax.text(4100, 2.5, 'K2', ha='center', backgroundcolor='white')
ax.text(18000, 8, 'Tess 1S', ha='center', backgroundcolor='white')
ax.text(65000, 2.5, 'K1 LC', ha='center', backgroundcolor='white')
ax.text(216000, 8, 'Tess 12S', ha='center', backgroundcolor='white')
ax.text(1950000, 2.5, 'K1 SC', ha='center', backgroundcolor='white')
plt.text(10000, 100, 'BLS')
plt.savefig('figure_speed.pdf', bbox_inches='tight')

# 4.25 yrs of Kepler K1 data at 1min (SC) and 30min (LC) cadence. K2 90 days at 30min.
# TESS 1 month: 18k obs at 2min cadence; 12 months https://arxiv.org/pdf/1809.07573.pdf
# CPU: Intel Core-i7 7700K, 4 cores, 8 threads, 4.5 GHz
# GPU: nVidia GTX1060
