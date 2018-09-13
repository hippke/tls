import numpy

y = numpy.array( [0.99, 1.01, 0.98, 0.90, 0.89, 1.0, 1.02, 1.01] )
y_in = y[3:5]
y_out = numpy.concatenate((y[0:3], y[5:8]))

dy = numpy.array( [0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.01, 0.01] )
dz = 1/dy**2
dy_out = numpy.concatenate((dz[0:3], dz[5:8]))
dy_in = dz[3:5]

signal = numpy.array( [1., 1., 1., 0.95, 0.95, 1., 1., 1.] )
signal_in = signal[3:5]
signal_out = numpy.concatenate((signal[0:3], signal[5:8]))


def outoftransit(y, dy, signal):    
    b1 = 0
    b2 = 0
    b4 = 0
    for i in range(len(y)):
        b1 = b1 + y[i] * dy[i]
        b2 = b2 + dy[i]
    b3 = b1 / b2
    for i in range(len(y)):
        b4 = b4 + (signal[i] - b3)**2 * dy[i]
    return -0.5 * b4


def intransit(y, dy, signal):
    a1 = 0
    a2 = 0
    for i in range(len(signal)):
        a1 = a1 + y[i] * dy[i]
        a2 = a2 + dy[i]
    a3 = a1 / a2
    a4 = 0
    print('a3', a3)
    for i in range(len(signal)):
        a4 = a4 + (signal_in[i] - a3)**2 * dy[i]
    print('-0.5 * a4', -0.5 * a4)
    return -0.5 * a4


a = intransit(y=y_in, dy=dy_in, signal=signal_in)
b = outoftransit(y=y_out, dy=dy_out, signal=signal_out)
ll = a + b
print(ll)
