
def T14(R_s, M_s, P):
        """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
           Output: Maximum planetary transit duration T_14max
                   Unit: Fraction of period P"""

        # astrophysical constants
        G = 6.673e-11  # [m^3 / kg / s^2]
        R_sun = 695508000  # [m]
        M_sun = 1.989*10**30  # [kg]

        P = P * 24 * 3600
        R_s = R_sun * R_s
        M_s = M_sun * M_s

        T14max = R_s * ((4*P) / (pi * G * M_s))**(1/3)
        result = T14max / P
        if result > 0.15:
            result = 0.15

        return result


def get_duration_grid(periods, log_step=None):
    if log_step is None:
        log_step = 1.1
    duration_max = T14(R_s=3.5, M_s=1.0, P=min(periods))
    duration_min = T14(R_s=0.13, M_s=0.1, P=max(periods))
    # Make grid
    durations = [duration_min]
    current_depth = duration_min
    while current_depth* log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint

    return durations


def get_depth_grid(y, deepest=None, shallowest=None, log_step=None):
    if deepest is None:
        deepest = 1 - numpy.min(y)
    if shallowest is None:
        shallowest = 10*10**-6  # 10 ppm
    if log_step is None:
        log_step = 1.1
    
    # Make grid
    depths = [shallowest]
    current_depth = shallowest
    while current_depth* log_step < deepest:
        current_depth = current_depth * log_step
        depths.append(current_depth)
    depths.append(deepest)  # Append endpoint

    return depths
