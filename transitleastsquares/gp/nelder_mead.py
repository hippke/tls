"""Nelder-Mead simplex minimisation, without a SciPy dependency.

A trimmed copy of `scipy.optimize._optimize._minimize_neldermead` (SciPy
1.18.1): bounds, callbacks, the adaptive coefficients and the `disp` /
`return_all` options are removed; everything else, including the order of
the floating-point operations, is unchanged, so the results are bit-identical
to `scipy.optimize.minimize(method="Nelder-Mead")` (tests/test_gp_nelder_mead.py).
This matters here: on multimodal likelihoods the path of Nelder-Mead decides
which optimum is found.

The copied code is under SciPy's license (BSD-3-Clause, also in
LICENSES/SCIPY.txt):

    Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np


class _MaxFev(Exception):
    pass


class Result:
    def __init__(self, x, fun, nfev, nit):
        self.x, self.fun, self.nfev, self.nit = x, fun, nfev, nit


def nelder_mead(func, x0, maxiter=None, maxfev=None, xatol=1e-4, fatol=1e-4, initial_simplex=None):
    """Minimise func(x) from x0 (options as in SciPy). Returns an object
    with x, fun, nfev (function evaluations) and nit (iterations)."""
    x0 = np.asarray(np.atleast_1d(x0).flatten(), dtype=float)
    rho, chi, psi, sigma = 1, 2, 0.5, 0.5
    N = len(x0)
    if initial_simplex is None:
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            y[k] = (1 + 0.05) * y[k] if y[k] != 0 else 0.00025
            sim[k + 1] = y
    else:
        sim = np.asarray(np.atleast_2d(initial_simplex), dtype=float).copy()
    maxfun = maxfev
    if maxiter is None and maxfun is None:
        maxiter = maxfun = N * 200
    elif maxiter is None:
        maxiter = N * 200 if maxfun == np.inf else np.inf
    elif maxfun is None:
        maxfun = N * 200 if maxiter == np.inf else np.inf
    ncalls = [0]

    def f(x):  # counts calls; the objective gets a copy of x, as in SciPy
        if ncalls[0] >= maxfun:
            raise _MaxFev
        ncalls[0] += 1
        fx = func(np.copy(x))
        return fx if np.isscalar(fx) else np.float64(np.asarray(fx).item())

    fsim = np.full((N + 1,), np.inf, dtype=float)
    try:
        for k in range(N + 1):
            fsim[k] = f(sim[k])
    except _MaxFev:
        pass
    ind = np.argsort(fsim)
    sim, fsim = np.take(sim, ind, 0), np.take(fsim, ind, 0)
    ind = np.argsort(fsim)
    fsim, sim = np.take(fsim, ind, 0), np.take(sim, ind, 0)
    iterations = 1
    while ncalls[0] < maxfun and iterations < maxiter:
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol
                    and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break
            xbar = np.add.reduce(sim[:-1], 0) / N  # centroid without the worst vertex
            xr = (1 + rho) * xbar - rho * sim[-1]  # reflection
            fxr = f(xr)
            doshrink = 0
            if fxr < fsim[0]:  # expansion
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                fxe = f(xe)
                if fxe < fxr:
                    sim[-1], fsim[-1] = xe, fxe
                else:
                    sim[-1], fsim[-1] = xr, fxr
            elif fxr < fsim[-2]:
                sim[-1], fsim[-1] = xr, fxr
            else:
                if fxr < fsim[-1]:  # outside contraction
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = f(xc)
                    if fxc <= fxr:
                        sim[-1], fsim[-1] = xc, fxc
                    else:
                        doshrink = 1
                else:  # inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = f(xcc)
                    if fxcc < fsim[-1]:
                        sim[-1], fsim[-1] = xcc, fxcc
                    else:
                        doshrink = 1
                if doshrink:  # shrink towards the best vertex
                    for j in range(1, N + 1):
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = f(sim[j])
            iterations += 1
        except _MaxFev:
            pass
        ind = np.argsort(fsim)
        sim, fsim = np.take(sim, ind, 0), np.take(fsim, ind, 0)
    return Result(sim[0], np.min(fsim), ncalls[0], iterations)
