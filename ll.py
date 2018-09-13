import numpy

y = numpy.array( [0.99, 1.01, 0.98, 0.90, 0.89, 1.0, 1.02, 1.01] )
y_in = y[3:5]
y_out = numpy.concatenate((y[0:3], y[5:8]))

dy = numpy.array( [0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.01, 0.01] )
dy_out = numpy.concatenate((dy[0:3], dy[5:8]))
dy_in = dy[3:5]

signal = numpy.array( [1., 1., 1., 0.95, 0.95, 1., 1., 1.] )
signal_in = signal[3:5]
signal_out = numpy.concatenate((signal[0:3], signal[5:8]))


# Out of transit
b1 = sum( y_out/dy_out**2 )
b2 = sum(1./dy_out**2)
#print(b)
b = b1 / b2
#print('b', b)
#print('signal_out', signal_out)
#print('dy_out', dy_out)
# correct: 
print('signal_out', signal_out)
bb = -1./2 * sum( (signal_out - b)**2 / dy_out**2)

#simplified:
#b = sum( (signal_out - 1)**2 / dy_out**2)
#bb = -1./2 * b
print('bb', bb)

# In transit
a1 = sum( y_in/dy_in**2 )
a2 = sum(1./dy_in**2)

a3 = a1 / a2
print('signal_in', signal_in)
print('a3', a3)
print('signal_in - a', signal_in - a3)
a4 = sum( (signal_in - a3)**2 / dy_in**2 )

a5 = -1./2 * a4
print(a5)

LL_1 = a5 + bb

print(LL_1)
