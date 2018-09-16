"""
I have an algorithm in Python which I like to be coded to run on a GPU with CUDA (e.g. using Tensorflow, PyCUDA, C++). The Python code is given below together with a short test. As this code will run many times with different data, I like to see how fast the speed can be on a GPU. Speed is what I am looking for.

- I expect you to have a CUDA GPU to test your results for speed
- Please contact me with an offer of how feasible this is, and how fast it could be
- If we complete this project with a good result, I have several more such functions for follow-up work.
- If the actual implementation is e.g. in C++, I would need a way to call it from Python
"""

# This is the function which shall be coded to run on a GPU with CUDA
@numba.njit(fastmath=True, parallel=False, cache=True)  
def function(data, signal, dy):
    result = numpy.zeros(len(data) - len(signal) + 1)
    for i in numba.prange(len(data) - len(signal) + 0):
        value = 0
        for j in numba.prange(len(signal) + 1):
            value = value + ((data[i+j]-signal[j])**2) * dy[i+j]
        result[i] = value
    return result
# End of algorithm. The code below is for calling this and testing it.

# Typical sizes for data and dy: 50000 floats FP32; signal: 1000 floats FP32
# The example below calls this function 100 times, which takes a total of 2 seconds on my machine. I could imagine that one would either
# - Run the interiors of the function in parallel
# - Alternatively, run the function as-is (like on a CPU), but parallelize the 100 subsequent function calls on the GPU
# - You are the specialist to test, recommend and implement the best solution

import time
import numpy
import numba

# Prepare test data
samples = 50000
no_of_signals = 100
fraction = 0.02  # Length of signal as fraction of sample size
numpy.random.seed(0)
data = numpy.random.normal(0, 1, samples)
dys = numpy.random.normal(0, 1, samples)
signals = numpy.random.normal(0, 1, int(samples * fraction))

for i in range(no_of_signals):
    signal = numpy.random.normal(0, 1, int(samples * fraction))
    signals = numpy.vstack([signals, signal])
    dy = numpy.random.normal(0, 1, int(samples))
    dys = numpy.vstack([dys, dy])

# Run once to compile the numba function, it is faster afterwards
function(data, signal, dy)

# Iterate over all signals
start = time.perf_counter()
for i in range(no_of_signals):
    my_result = function(data, signals[i], dys[i])
stop = time.perf_counter()
if round(sum(my_result)) == -196546:
    print('Calculation correct')
else:
    print('Calculation incorrect')

print('Time (sec)', stop-start)
