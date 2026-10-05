"""Command line interface: transitleastsquares <lightcurve.csv> [-o DIR] [-c CONFIG]"""

import argparse
import os
from configparser import ConfigParser

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.main import transitleastsquares


def read_config(filename):
    """Return (power kwargs, delimiter) from a TLS config file (see tls_config.cfg)"""
    config = ConfigParser()
    if not config.read(filename):
        raise OSError(f"Cannot read {filename}")
    grid, speed = config["Grid"], config["Speed"]
    kwargs = dict(
        R_star=float(grid["R_star"]),
        R_star_min=float(grid["R_star_min"]),
        R_star_max=float(grid["R_star_max"]),
        M_star=float(grid["M_star"]),
        M_star_min=float(grid["M_star_min"]),
        M_star_max=float(grid["M_star_max"]),
        period_min=float(grid["period_min"]),
        period_max=float(grid["period_max"]),
        n_transits_min=int(grid["n_transits_min"]),
        transit_template=config["Template"]["transit_template"],
        duration_grid_step=float(speed["duration_grid_step"]),
        transit_depth_min=float(speed["transit_depth_min"]),
        oversampling_factor=int(speed["oversampling_factor"]),
        T0_fit_margin=float(speed["T0_fit_margin"]),
        use_threads=int(speed["use_threads"]),
    )
    return kwargs, config["File"]["delimiter"]


def main(argv=None):
    print(tls_constants.TLS_VERSION)
    parser = argparse.ArgumentParser()
    parser.add_argument("lightcurve", help="path to lightcurve file")
    parser.add_argument("-o", "--output", help="path to output directory")
    parser.add_argument("-c", "--config", help="path to configuration file")
    args = parser.parse_args(argv)

    kwargs, delimiter = {}, ","
    if args.config is not None:
        try:
            kwargs, delimiter = read_config(args.config)
            print("Using TLS configuration from config file", args.config)
        except (OSError, KeyError, ValueError):
            print(
                "Using default values because of broken or missing configuration file",
                args.config,
            )
    else:
        print("No config file given. Using default values")

    data = numpy.genfromtxt(args.lightcurve, delimiter=delimiter)
    t, y = data[:, 0], data[:, 1]
    dy = data[:, 2] if data.shape[1] > 2 else None
    results = transitleastsquares(t, y, dy).power(**kwargs)

    # Determine path and file names of output files
    base = args.lightcurve
    if args.output is not None:
        base = os.path.join(args.output, os.path.basename(args.lightcurve))
    file_stats, file_power = base + "_statistics.csv", base + "_power.csv"

    try:
        numpy.savetxt(
            file_power,
            numpy.column_stack([results.periods, results.power]),
            delimiter=",",
            fmt="%1.6f",
        )
        print("SDE-ogram saved to", file_power)

        statistics = dict(list(results.items())[0:28])  # scalar + per-transit stats
        numpy.set_printoptions(precision=8, threshold=10e10)
        with open(file_stats, "w") as f:
            for key, value in statistics.items():
                f.write(f"{key} {value}\n")
        print("Statistics saved to", file_stats)
    except OSError:
        print("Error saving result file")


if __name__ == "__main__":
    main()
