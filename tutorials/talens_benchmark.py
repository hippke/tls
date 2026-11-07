"""Injection and recovery on the simulated PLATO light curves of Talens et al. (2026),
with TLS 2.0 + GP (transitleastsquares.gp) in the setting of tutorial 14.

Data (Oxford University Research Archive, doi:10.5287/ora-aeaxy98vd): unzip
full-noplanet.zip (1.5 GB) into DATA_DIR and put star_catalog.ftr,
planet_catalog_inner.ftr and planet_catalog_outer.ftr next to it, so that
DATA_DIR/lightcurves/full_noplanet/cadence_600s/0000000000.ftr exists.
Needs pandas, pyarrow (to read .ftr files) and batman-package (injection).

Usage (the full benchmark: 2000 + 2000 injections and 500 planet-free light curves):
    python talens_benchmark.py DATA_DIR results.jsonl [--n N] [--device gpu] [--workers 8]
    python talens_benchmark.py DATA_DIR results.jsonl --summary
--n N: only the first N stars of each sample. Finished searches in results.jsonl
are skipped, so an interrupted run continues where it stopped.
"""

import argparse
import json
import math
import multiprocessing
import os
import time
import traceback
import warnings

import numpy as np

CADENCE = 600.0 / 86400  # days
N_STARS = {"inner": 2000, "outer": 2000, "noise": 500}

# Published recoveries without a threshold (Talens et al. 2026, Tables A.1, A.2): percent of 2000
PUBLISHED = {
    "nuance": (68.7, 53.8),
    "CETRA, best Huber spline": (65.5, 46.1),
    "umbra, best Huber spline": (66.3, 50.5),
    "CETRA, six Huber splines": (69.6, 56.1),
    "umbra, six Huber splines": (71.6, 59.7),
}


def settings(R, M):
    """gp.search options of the benchmark for a star of radius R and mass M (solar units)."""
    return dict(
        period_min=5.0,  # as nuance in Talens et al.
        n_transits_min=2,  # at least two transits with data ...
        grid_transits_min=1,  # ... at all periods up to the time span
        segment_gap=0.5,  # the quarters: each its own filter and free mean
        segment_degree=4,  # and a 4th-order polynomial per quarter (as nuance's mean model)
        R_star_min=0.8 * R, R_star_max=1.2 * R,  # +-20 % stellar limits (as Talens et al.)
        M_star_min=0.8 * M, M_star_max=1.2 * M,
        duration_min_scale=0.44,  # shortest duration for impact parameters up to 0.9: sqrt(1 - 0.9^2)
        trend=False,
    )


def stars(data_dir):
    import pandas as pd

    return pd.read_feather(os.path.join(data_dir, "star_catalog.ftr"))


def planets(data_dir, sample):
    import pandas as pd

    return pd.read_feather(os.path.join(data_dir, f"planet_catalog_{sample}.ftr"))


def load(data_dir, star):
    """Planet-free light curve of a star: time (days), flux and error normalised to the median."""
    import pandas as pd

    d = pd.read_feather(os.path.join(data_dir, "lightcurves", "full_noplanet", "cadence_600s", f"{star * 100:010d}.ftr"))
    t, f, e = (np.asarray(d[k], float) for k in ("time", "flux", "flux_err"))
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    m = np.median(f[ok])
    return t[ok] / 86400.0, f[ok] / m, e[ok] / m


def t14(row):
    """Total transit duration (days), circular orbit."""
    a, k, b, P = row["ap_Rstar"], row["Rp_Rstar"], row["impact_parameter"], row["Period"]
    x = (1 + k) ** 2 - b**2
    if x <= 0:
        return 0.0
    inc = math.radians(row["inclination"])
    return P / math.pi * math.asin(min(1.0, math.sqrt(x) / (a * math.sin(inc))))


def inject(t, f, row):
    """The planet of the catalogue row with batman: nonlinear limb darkening, integrated over the 600 s exposures."""
    import batman

    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = row["T0"], row["Period"], row["Rp_Rstar"], row["ap_Rstar"], row["inclination"]
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark = "nonlinear"
    p.u = [row["LD_a1"], row["LD_a2"], row["LD_a3"], row["LD_a4"]]
    m = batman.TransitModel(p, t, supersample_factor=11, exp_time=CADENCE, fac=1e-3)  # fixed step: always converges
    return f * m.light_curve(p)


def transits_with_data(t, row, dur):
    """Mid-transit times of the injected planet that have data within half a duration."""
    P, T0 = row["Period"], row["T0"]
    n0, n1 = math.floor((t.min() - T0) / P) - 1, math.ceil((t.max() - T0) / P) + 1
    return [T0 + n * P for n in range(n0, n1 + 1) if np.any(np.abs(t - (T0 + n * P)) < 0.5 * max(dur, CADENCE))]


def recovered(inj, res, sample):
    """Talens et al. Eq. (4): the period or a multiple of 1/2, 2, 3 of it accumulates a timing error of less
    than T14 / 2 over the transits; Eq. (5), outer sample with one or two transits: the recovered T0 lies within
    T14 / 2 of an injected transit."""
    P, T14, ntr = inj["P"], inj["T14"], max(inj["n_tr"], 1)
    if any(abs(P - n * res["period"]) * ntr < T14 / 2 for n in (0.5, 1, 2, 3)):
        return True
    if sample == "outer" and inj["n_tr"] <= 2 and inj["transits"]:
        return min(abs(res["T0"] - tc) for tc in inj["transits"]) < T14 / 2
    return False


def experiment(data_dir, sample, star, device="cpu", use_threads=None):
    """One search: inject (unless sample is "noise"), search, and return a record (dict)."""
    from transitleastsquares import gp

    warnings.simplefilter("ignore")
    rec = dict(sample=sample, star=star)
    t_start = time.perf_counter()
    try:
        st = stars(data_dir).iloc[star]
        t, f, e = load(data_dir, star)
        if sample != "noise":
            row = planets(data_dir, sample).iloc[star]
            f = inject(t, f, row)
            dur = t14(row)
            tr = transits_with_data(t, row, dur)
            rec["inj"] = dict(P=float(row["Period"]), T0=float(row["T0"]), Rp_Rstar=float(row["Rp_Rstar"]),
                              R_planet=float(row["Rp_Rstar"] * st["Radius"] * 109.1), T14=dur, transits=tr, n_tr=len(tr))
        r = gp.search(t, f, e, device=device, use_threads=use_threads, **settings(st["Radius"], st["Mass"]))
        rec.update(period=float(r.period), T0=float(r.T0), duration=float(r.duration), depth=float(r.depth),
                   SDE=float(r.SDE), snr=float(r.snr))
        if sample != "noise":
            rec["recovered"] = bool(recovered(rec["inj"], rec, sample))
    except Exception as ex:  # noqa: BLE001
        rec["error"] = repr(ex)
        rec["trace"] = traceback.format_exc()[-1500:]
    rec["time"] = time.perf_counter() - t_start
    return rec


def tasks(n=None):
    """(sample, star) of all experiments, the first n stars of each sample if n is given."""
    return [(s, k) for s in ("inner", "outer", "noise") for k in range(min(n or N_STARS[s], N_STARS[s]))]


def done(out):
    if not os.path.exists(out):
        return {}
    recs = {}
    for line in open(out):
        try:
            r = json.loads(line)
        except ValueError:  # a line being written
            continue
        if "error" not in r:
            recs[(r["sample"], r["star"])] = r
    return recs


def _work(args):
    return experiment(*args)


def run(data_dir, out, n=None, device="cpu", workers=1, verbose=True):
    """Run the experiments not yet in out (JSON lines), in parallel processes if workers > 1."""
    old = done(out)
    todo = [x for x in tasks(n) if x not in old]
    if verbose:
        print(f"{len(todo)} experiments to run ({len(old)} done)", flush=True)
    if not todo:
        return
    with open(out, "a") as fh:
        def write(r):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if verbose:
                msg = r.get("error") or (f"P = {r['period']:.3f} d, SDE {r['SDE']:.1f}" +
                                         (f", recovered: {r['recovered']}" if "recovered" in r else ""))
                print(f"{r['sample']:5s} star {r['star']:4d}: {msg} ({r['time']:.1f} s)", flush=True)

        write(experiment(data_dir, *todo[0], device=device))  # the first also compiles the code (numba cache)
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                for r in pool.imap_unordered(_work, [(data_dir, s, k, device, 1) for s, k in todo[1:]]):
                    write(r)
        else:
            for s, k in todo[1:]:
                write(experiment(data_dir, s, k, device=device))


def summary(out, fap=0.01):
    """Recovered percent per sample without a threshold and at the false-alarm rate fap (SDE threshold from
    the planet-free light curves), as a dict; and the published values for comparison."""
    recs = list(done(out).values())
    noise = np.array([r["SDE"] for r in recs if r["sample"] == "noise"])
    thr = float(np.quantile(noise, 1 - fap)) if len(noise) >= 20 else None
    res = dict(threshold=thr, n_noise=len(noise))
    for s in ("inner", "outer"):
        x = [r for r in recs if r["sample"] == s]
        if not x:
            continue
        f = np.array([r["recovered"] for r in x])
        res[s] = dict(n=len(x), recovered=100 * f.mean(), err=100 * math.sqrt(f.mean() * (1 - f.mean()) / len(x)))
        if thr is not None:
            res[s]["recovered_fap"] = 100 * np.mean(f & (np.array([r["SDE"] for r in x]) >= thr))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, help="only the first N stars of each sample")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (several can share one GPU)")
    ap.add_argument("--summary", action="store_true", help="only print the recoveries of OUT")
    a = ap.parse_args()
    if not a.summary:
        run(a.data_dir, a.out, a.n, a.device, a.workers)
    s = summary(a.out)
    for k in ("inner", "outer"):
        if k in s:
            line = f"{k}: {s[k]['recovered']:.1f} +- {s[k]['err']:.1f} % of {s[k]['n']} without a threshold"
            if "recovered_fap" in s[k]:
                line += f", {s[k]['recovered_fap']:.1f} % at 1 % false alarms (SDE >= {s['threshold']:.1f}, {s['n_noise']} planet-free)"
            print(line)
    print("Talens et al. (2026), without a threshold (inner, outer):")
    for k, v in PUBLISHED.items():
        print(f"  {k}: {v[0]} %, {v[1]} %")


if __name__ == "__main__":
    main()
