import numpy
import numba
import time


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def out_of_transit_residuals(data, width_signal, dy):
    outer_loop_lenght = len(data) - width_signal + 1
    inner_loop_length = len(data) - width_signal
    chi2 = numpy.zeros(outer_loop_lenght)
    for i in numba.prange(outer_loop_lenght):
        value = 0
        start_transit = i
        end_transit = i + width_signal
        for j in numba.prange(inner_loop_length):
            #if j < start_transit or j > end_transit:
            if (end_transit > j) or (j < start_transit):
                value = value + (1 - data[j])#**2# * dy[j]
        chi2[i] = value
        #print(i, value)
    return chi2


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def ootr(data, width_signal, dy):
    chi2 = numpy.zeros(len(data) - width_signal + 1)
    fullsum = numpy.sum((1 - data)**2 * dy)
    window = numpy.sum((1 - data[:width_signal])**2 * dy[:width_signal])
    chi2[0] = fullsum-window
    for i in range(1, len(data) - width_signal + 1):
        drop_first = (1 - data[i-1])**2 * dy[i-1]
        add_next = (1 - data[i + width_signal-1])**2 * dy[i+width_signal-1]
        chi2[i] = chi2[i-1] - drop_first + add_next
    return chi2



size = 50000
fraction = 0.02
data = numpy.full(size, 0.99)
dy = 1/numpy.full(size, 0.01)
width_signal = int(size * fraction)

result = ootr(data, width_signal, dy)
t1 = time.perf_counter()
result = ootr(data, width_signal, dy)
t2 = time.perf_counter()
#print(result)
#print('Sum', numpy.sum(result))
ttt = t2-t1
print('Time new', format(ttt, '.5f'))

result = out_of_transit_residuals(data, width_signal, dy)
t1 = time.perf_counter()
result = out_of_transit_residuals(data, width_signal, dy)
t2 = time.perf_counter()
#print(result)
#print('Sum', numpy.sum(result))
print('Time old', format(t2-t1, '.5f'), 'Speedup', format((t2-t1)/ttt, '.5f'))
