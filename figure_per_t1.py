import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.2, 1000)
plt.ylim(0.0003, 0.2)
ax.scatter(data['per'], data['t14'], s=10, alpha=0.5, linewidths=0.5, edgecolor='black')
plt.plot((0.3, 1000), (3.5*10**-2, 1*10**-4), color='black', linestyle='dashed', linewidth=1)
plt.plot((3, 1000), (0.15, 2.5*10**-3), color='black', linestyle='dashed', linewidth=1)
plt.plot((0.3, 3), (0.15, 0.15), color='black', linestyle='dashed', linewidth=1)
plt.plot((0.3, 0.3), (0.15, 0.035), color='black', linestyle='dashed', linewidth=1)

ax.set_xlabel(r'Period (days)')
ax.set_ylabel(r'Transit duration ($T_{\rm 14}/P$)')
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
plt.savefig('figure_per_t14.pdf', bbox_inches='tight')
