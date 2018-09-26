import numpy
from TransitLeastSquares import ootr_efficient, in_transit_residuals


data = numpy.array((1, 1.01,0.99,0.9,0.89,1.01,1,1.01))
dy = numpy.array((0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01))
dy = 1/dy**2
signal = numpy.array((0.9, 0.9))
signal_width = numpy.size(signal)

itr = in_transit_residuals(data, signal, dy)
ootr = ootr_efficient(data, signal_width, dy)
print(itr)
print(ootr)
print(itr+ootr)

diff1 = itr - [221., 202.,  81.,   1., 122., 221., 221.]
if sum(diff1) > 0.001:
    print('In-transit calculation incorrect')
else:
    print('OK: In-transit calculation')

diff2 = ootr - [224., 223., 124.,   4., 103., 224., 224.]
if sum(diff2) > 0.001:
    print('Out-of-transit calculation incorrect')
else:
    print('OK: Out-of-transit calculation')
