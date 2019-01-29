import numpy
from os import path
import transitleastsquares.tls_constants as tls_constants
from transitleastsquares.helpers import running_median


def FAP(SDE):
    """Returns FAP (False Alarm Probability) for a given SDE"""
    data = numpy.genfromtxt(
        path.join(tls_constants.resources_dir, "fap.csv"),
        dtype="f8, f8",
        delimiter=",",
        names=["FAP", "SDE"],
    )
    return data["FAP"][numpy.argmax(data["SDE"] > SDE)]


def rp_rs_from_depth(depth, law, params):
    """Takes the maximum transit depth, limb-darkening law and parameters
    Returns R_P / R_S (ratio of planetary to stellar radius)
    Source: Heller 2019, https://arxiv.org/abs/1901.01730"""

    # Validations:
    # - LD law must exist
    # - All parameters must be floats or ints
    # - All parameters must be given in the correct quanitity for the law

    if len(params) == 1:
        params = float(params[0])

    if not isinstance(params, (float, int)) and not all(
        isinstance(x, (float, int)) for x in params
    ):
        raise ValueError("All limb-darkening parameters must be numbers")

    laws = "linear, quadratic, squareroot, logarithmic, nonlinear"
    if law not in laws:
        raise ValueError("Please provide a supported limb-darkening law:", laws)

    if law == "linear" and not isinstance(params, float):
        raise ValueError("Please provide exactly one parameter")

    if law in "quadratic, logarithmic, squareroot" and len(params) != 2:
        raise ValueError("Please provide exactly two limb-darkening parameters")

    if law == "nonlinear" and len(params) != 4:
        raise ValueError("Please provide exactly four limb-darkening parameters")

    # Actual calculations of the return value
    if law == "linear":
        return (depth * (1 - params / 3)) ** (1 / 2)

    if law == "quadratic":
        return (depth * (1 - params[0] / 3 - params[1] / 6)) ** (1 / 2)

    if law == "squareroot":
        return (depth * (1 - params[0] / 3 - params[1] / 5)) ** (1 / 2)

    if law == "logarithmic":
        return (depth * (1 + 2 * params[1] / 9 - params[0] / 3)) ** (1 / 2)

    if law == "nonlinear":
        return (
            depth
            * (1 - params[0] / 5 - params[1] / 3 - 3 * params[2] / 7 - params[3] / 2)
        ) ** (1 / 2)


def pink_noise(data, width):
    std = 0
    datapoints = len(data) - width + 1
    for i in range(datapoints):
        std += numpy.std(data[i : i + width]) / width ** 0.5
    return std / datapoints


def period_uncertainty(periods, power):
    # Determine estimate for uncertainty in period
        # Method: Full width at half maximum
    try:
        # Upper limit
        index_highest_power = numpy.argmax(power)
        idx = index_highest_power
        while True:
            idx += 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_upper = idx
                break
        # Lower limit
        idx = index_highest_power
        while True:
            idx -= 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_lower = idx
                break
        period_uncertainty = 0.5 * (
            periods[idx_upper] - periods[idx_lower]
        )
    except:
        period_uncertainty = float("inf")
    return period_uncertainty


def spectra(chi2, oversampling_factor):
    SR = numpy.min(chi2) / chi2
    SDE_raw = (1 - numpy.mean(SR)) / numpy.std(SR)

    # Scale SDE_power from 0 to SDE_raw
    power_raw = SR - numpy.mean(SR)  # shift down to the mean being zero
    scale = SDE_raw / numpy.max(power_raw)  # scale factor to touch max=SDE_raw
    power_raw = power_raw * scale

    # Detrended SDE, named "power"
    kernel = oversampling_factor * tls_constants.SDE_MEDIAN_KERNEL_SIZE
    if kernel % 2 == 0:
        kernel = kernel + 1
    if len(power_raw) > 2 * kernel:
        my_median = running_median(power_raw, kernel)
        power = power_raw - my_median
        # Re-normalize to range between median = 0 and peak = SDE
        # shift down to the mean being zero
        power = power - numpy.mean(power)
        SDE = numpy.max(power / numpy.std(power))
        # scale factor to touch max=SDE
        scale = SDE / numpy.max(power)
        power = power * scale
    else:
        power = power_raw
        SDE = SDE_raw

    return SR, power, power_raw, SDE_raw, SDE

