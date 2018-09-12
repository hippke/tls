import time
import numpy
import numba



@numba.njit(fastmath=True, parallel=False, cache=True)  
def itr(data, signal, dy, ootr):
    width_signal = signal.shape[0]
    width_data = data.shape[0]
    result = numpy.zeros(width_data - width_signal + 1)
    for i in numba.prange(width_data - width_signal + 1):
        value = 0
        for j in range(width_signal):
            value = value + ((data[i+j]-signal[j])**2) * dy[i+j]
        result[i] = value + ootr[i]
    return result


# Prepare test data
samples = 32000
no_of_signals = 100
fraction = 0.01
data = numpy.random.normal(0, 1, samples)
signals = numpy.random.normal(0, 1, int(samples * fraction))
dys = numpy.random.normal(0, 1, int(samples * fraction))
ootr = numpy.random.normal(0, 1, int(samples * fraction))
for i in range(no_of_signals):
    signal = numpy.random.normal(0, 1, int(samples * fraction))
    signals = numpy.vstack([signals, signal])
    dy = numpy.random.normal(0, 1, int(samples * fraction))
    dys = numpy.vstack([dys, dy])

# Run once to compile the numba part
itr(data, signal, dy, ootr)
start = time.perf_counter()

# Iterate over all signals
for i in range(no_of_signals):
    itr(data, signal, dy, ootr)
stop = time.perf_counter()
print('Samples:', samples, '', stop-start)
