import numpy


class TransitLeastSquares(object):
    def power(parameters, t, y, dy=None):

        """Validate parameters and set default values for those missing"""

        # Data
        duration = max(t) - min(t)
        if duration <= 0:
            raise ValueError('Time duration must positive')
        if min(y) < 0:
            raise ValueError('Flux values must be positive')
        if max(y) >= float('inf'):
            raise ValueError('Flux values must be finite')
        # If no dy is given, create it with the standard deviation of the flux
        if dy is None:
            dy = numpy.full(len(y), numpy.std(y))
        if numpy.size(t) != numpy.size(y) or numpy.size(t) != numpy.size(dy):
            raise ValueError('Arrays (t, y, dy) must be of the same dimensions')
        if t.ndim != 1:  # Size identity ensures dimensional identity
            raise ValueError('Inputs (t, y, dy) must be 1-dimensional')


        # Stellar radius
        # (0 < R_star < inf) must exist 
        if not hasattr(parameters, 'R_star'):
            parameters.R_star = 1.0
        if parameters.R_star <= 0 or parameters.R_star >= float('inf'):
            raise ValueError('R_star must be positive')

        # Assert (0 < R_star_min <= R_star)
        if not hasattr(parameters, 'R_star_min'):
            parameters.R_star_min = 0.13
        if parameters.R_star_min > parameters.R_star:
            raise ValueError('R_star_min <= R_star is required')
        if parameters.R_star_min <= 0 or parameters.R_star_min >= float('inf'):
            raise ValueError('R_star_min must be positive')

        # Assert (R_star <= R_star_max < inf)
        if not hasattr(parameters, 'R_star_max'):
            parameters.R_star_max = 3.5
        if parameters.R_star_max < parameters.R_star:
            raise ValueError('R_star_max >= R_star is required')
        if parameters.R_star_max <= 0 or parameters.R_star_max >= float('inf'):
            raise ValueError('R_star_max must be positive')


        # Stellar mass
        # Assert (0 < M_star < inf)
        if not hasattr(parameters, 'M_star'):
            parameters.M_star = 1.0
        if parameters.M_star <= 0 or parameters.M_star >= float('inf'):
            raise ValueError('M_star must be positive')

        # Assert (0 < M_star_min <= M_star)
        if not hasattr(parameters, 'M_star_min'):
            parameters.M_star_min = 0.1
        if parameters.M_star_min > parameters.M_star:
            raise ValueError('M_star_min <= M_star is required')
        if parameters.M_star_min <= 0 or parameters.M_star_min >= float('inf'):
            raise ValueError('M_star_min must be positive')

        # Assert (M_star <= M_star_max < inf)
        if not hasattr(parameters, 'M_star_max'):
            parameters.M_star_max = 1.0
        if parameters.M_star_max < parameters.M_star:
            raise ValueError('M_star_max >= R_star required')
        if parameters.M_star_max <= 0 or parameters.M_star_max >= float('inf'):
            raise ValueError('M_star_max must be positive')


        # Period grid
        if not hasattr(parameters, 'period_min'):
            parameters.period_min = 0
        if not hasattr(parameters, 'period_max'):
            parameters.period_max = float('inf')
        if parameters.period_min < 0:
            raise ValueError('period_min >= 0 required')
        if parameters.period_min >= parameters.period_max:
            raise ValueError('period_min < period_max required')
        if not hasattr(parameters, 'n_transits_min'):
            parameters.n_transits_min = 2
        if not isinstance(parameters.n_transits_min, int):
            raise ValueError('n_transits_min must be an integer value')
        if parameters.n_transits_min < 1:
            raise ValueError('n_transits_min must be an integer value >= 1')







        return t*y



class parameters(object):
    def __init__(self):
        pass


parameters.R_star = 1
parameters.n_transits_min = 5
#parameters.R_starMin = 0.2
#t = y = 2
t = numpy.linspace(1, 100, 100)
y = t
print(parameters.R_star)
results = TransitLeastSquares.power(parameters, t, y)
#print(results)



"""

Physical parameters of the fitted transit:
:per: *(float)* Orbital period (in units of days). Default: X. Optional.
:rp: *(float)* Planet radius (in units of stellar radii). Default: X. Optional.
:a: *(float)* Semi-major axis (in units of stellar radii). Default: X. Optional.
:inc: *(float)* Orbital inclination (in degrees). Default: 90. Optional.
:b: *(float)* Orbital impact parameter as the sky-projected distance between the centre of the stellar disc and the centre of the planetary disc at conjunction. If set, overrules ``inc=degrees(arccos(b/a)``. Default: 0. Optional.
:ecc: *(float)* Orbital eccentricity. Default: 0. Optional.
:w: *(float)* Argument of periapse (in degrees). Default: 90. Optional.
:u: *(array)* List of limb darkening coefficients. Default: [X, Y]. Optional.
:limb_dark: *(str)* Limb darkening model (choice of ``nonlinear``, ``quadratic``, ``exponential``, ``logarithmic``, ``squareroot``, ``linear``, ``uniform``, ``power2``, or ``custom``). Default: ``quadratic``. Optional.


Parameters used to balance detection efficiency and computational requirements:
:duration_grid_step: *(float)* Grid step width between subsequent trial durations, so that :math:`{\rm dur}_{n+1}={\rm dur}_n \times {\rm duration_grid_step}`. Default: 1.1 (i.e., each subsequent trial duration is longer by 10%). Optional.
:transit_depth_min: *(float)* Shallowest transit depth to be fitted. Transit depths down to half the transit_depth_min can be found at reduced sensitivity. A reasonable value should be estimated from the data to balance sensitivity and avoid fitting the noise floor. Overfitting may cause computational requirements larger by a factor of 10. For reference, the shallowest transit from Kepler is 11.9 ppm (Kepler-37b, `Barclay et al. 2013 <http://adsabs.harvard.edu/abs/2013Natur.494..452B>`_). Default: 10 ppm. Optional.
:oversampling_factor: *(int)* Oversampling of the period grid to avoid that the true period falls in between trial periods and is missed. Default: 2. Optional.

"""
