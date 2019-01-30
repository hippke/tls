import numpy
from transitleastsquares import cleaned_array

if __name__ == "__main__":
    print("Starting test: cleaned_array...", end='')
    
    dirty_array = numpy.ones(10, dtype=object)
    time_array = numpy.linspace(1, 10, 10)
    dy_array = numpy.ones(10, dtype=object)
    dirty_array[1] = None
    dirty_array[2] = numpy.inf
    dirty_array[3] = -numpy.inf
    dirty_array[4] = numpy.nan
    dirty_array[5] = -99

    t, y, dy = cleaned_array(time_array, dirty_array, dy_array)
    numpy.testing.assert_equal(len(t), 5)
    numpy.testing.assert_equal(numpy.sum(t), 35)
    print('passed')
    