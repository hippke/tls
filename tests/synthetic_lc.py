"""batman-like interface to TLS' own transit model, for generating synthetic
test light curves without the batman dependency."""

from transitleastsquares.transit_model import light_curve


class TransitParams:
    t0 = 0.0
    per = 1.0
    rp = 0.1
    a = 10.0
    inc = 90.0
    ecc = 0.0
    w = 90.0
    u = ()
    limb_dark = "quadratic"


class TransitModel:
    def __init__(self, params, t):
        self.t = t

    def light_curve(self, p):
        return light_curve(
            self.t, p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.u, p.limb_dark
        )
