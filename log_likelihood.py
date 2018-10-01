# From https://arxiv.org/pdf/1809.11116.pdf

import numpy
from numpy import log

flux  =  numpy.array((1.01, 1.00, 1.01, 0.99, 1.00, 0.95, 0.90, 0.89, 0.96, 1.01, 0.99))
model1 = numpy.array((1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.00))
model2 = numpy.array((1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00))
error  = numpy.array((0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03))

logL1 = numpy.sum(-0.5 * ((flux-model1)**2 / error**2) - 2 * log(error**2))
logL2 = numpy.sum(-0.5 * ((flux-model2)**2 / error**2) - 2 * log(error**2))
print(logL1)
print(logL2)
