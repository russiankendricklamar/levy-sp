"""
products.py — pricers for the four structured products used in the thesis.

European products (capital-protected note, digital call) are priced by the
Carr-Madan FFT (exact, all 7 models). Path-dependent products (barrier reverse
convertible, Phoenix autocall) are priced by Monte Carlo on simulated paths
(6 models; CGMY has no path simulator, European only).

All pricers assume a notional of 100 and take a calibrated-parameter dict `p`
plus market inputs S0, r, q. Model dispatch goes through svmodels.price_call_any
(FFT) and simulators.simulate (paths).

Author: master's-thesis codebase. Reproducible on real calibrated params.
"""
import numpy as np
import svmodels
import simulators


# ---------------------------------------------------------------------------
# European: FFT-priced
# ---------------------------------------------------------------------------

def price_call(K, T, S0, r, q, model, p):
    """European call price via Carr-Madan FFT (any of the 7 models)."""
    return float(svmodels.price_call_any(K, T, S0, r, q, model, p))


def note_participation(S0, r, q, model, p, T=3.0, notional=100.0):
    """
    Capital-protected note (default 3y): pays notional back at T plus
    `participation` x positive index return. Returns the fair participation
    rate (%) that makes the note price equal par (=notional).

    note = PV(notional) + participation x (ATM-call value per unit notional)
    set note = notional => participation = (notional - PV(notional)) / atm_call_notional
    """
    disc = np.exp(-r * T)
    atm_call = price_call(S0, T, S0, r, q, model, p)       # strike = spot
    atm_call_notional = notional * atm_call / S0            # payoff scaled to notional
    part = (notional - notional * disc) / atm_call_notional
    return 100.0 * part


def digital_call(S0, r, q, model, p, T=1.0, payout=10.0, eps=None):
    """
    Digital (binary) ATM call (default 1y): pays `payout` if S_T >= S0.
    Priced as the negative strike-derivative of the call price
    (central finite difference), consistent with the FFT engine.
    """
    if eps is None:
        eps = S0 * 1e-3
    cu = price_call(S0 + eps, T, S0, r, q, model, p)
    cd = price_call(S0 - eps, T, S0, r, q, model, p)
    dCdK = (cu - cd) / (2 * eps)
    disc = np.exp(-r * T)
    # P(S_T >= K) under Q = -e^{rT} dC/dK
    prob = -np.exp(r * T) * dCdK
    return float(payout * disc * prob)


# ---------------------------------------------------------------------------
# Path-dependent: Monte-Carlo priced
# ---------------------------------------------------------------------------

def _paths(model, p, S0, r, q, T, n_steps, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    return simulators.simulate(model, p, S0, r, q, T, n_steps, n_paths, rng)


def price_brc(coupon, S0, r, q, model, p, T=1.0, ki=0.70,
             n_steps=252, n_paths=40000, seed=0, notional=100.0):
    """
    Barrier Reverse Convertible (default 1y, 70% knock-in).
    Investor receives `coupon` (annualized, %) regardless; principal is
    protected UNLESS the daily-monitored barrier ki*S0 is breached, in which
    case redemption tracks the (capped-at-par) index performance.
    Returns the present value (per notional=100) for a given coupon.
    """
    S = _paths(model, p, S0, r, q, T, n_steps, n_paths, seed)
    ST = S[:, -1]
    min_path = S.min(axis=1)
    breached = min_path <= ki * S0
    redemption = np.where(~breached | (ST >= S0), notional, notional * ST / S0)
    disc = np.exp(-r * T)
    redemption_pv = disc * redemption.mean()
    coupon = (notional - redemption_pv) / (disc * notional * T) * 100.0
    return float(coupon)


def fair_coupon_brc(S0, r, q, model, p, T=1.0, ki=0.70,
                   n_steps=252, n_paths=40000, seed=0, notional=100.0):
    """Fair annualised coupon (%) that makes BRC price = par."""
    return price_brc(0.0, S0, r, q, model, p, T, ki, n_steps, n_paths, seed, notional)


def price_autocall(coupon, S0, r, q, model, p, T=3.0, obs_per_year=4,
                  ac=1.00, cb=0.70, pb=0.60, memory=True,
                  n_steps=252 * 3, n_paths=40000, seed=0, notional=100.0):
    """
    Phoenix autocall (default 3y, quarterly obs, autocall @100%, coupon
    barrier 70%, protection barrier 60%, memory coupons).
    `coupon` is the per-period coupon in % of notional. Returns present value.
    """
    n_obs = int(round(T * obs_per_year))
    obs_idx = np.linspace(n_steps / n_obs, n_steps, n_obs).round().astype(int)
    S = _paths(model, p, S0, r, q, T, n_steps, n_paths, seed)
    dt_obs = T / n_obs
    N = S.shape[0]
    alive = np.ones(N, dtype=bool)
    pv = np.zeros(N)
    missed = np.zeros(N)          # memory bucket
    cpn = (coupon / 100.0) * notional
    for k, idx in enumerate(obs_idx, start=1):
        t = k * dt_obs
        disc = np.exp(-r * t)
        Sk = S[:, idx - 1]
        pay_cpn = alive & (Sk >= cb * S0)
        if memory:
            n_pay = np.where(pay_cpn, missed + 1, 0)
            pv += np.where(pay_cpn, disc * cpn * n_pay, 0.0)
            missed = np.where(pay_cpn, 0, np.where(alive, missed + 1, missed))
        else:
            pv += np.where(pay_cpn, disc * cpn, 0.0)
        # autocall before maturity
        called = alive & (Sk >= ac * S0) & (k < n_obs)
        pv += np.where(called, disc * notional, 0.0)
        alive = alive & ~called
    # maturity redemption for still-alive paths
    ST = S[:, -1]
    disc_T = np.exp(-r * T)
    redemption = np.where(ST >= pb * S0, notional, ST / S0 * notional)
    pv += np.where(alive, disc_T * redemption, 0.0)
    return float(pv.mean())


def fair_coupon_ac(S0, r, q, model, p, T=3.0, obs_per_year=4,
                  ac=1.00, cb=0.70, pb=0.60, memory=True,
                  n_steps=252 * 3, n_paths=40000, seed=0, notional=100.0,
                  tol=1e-3, lo=0.0, hi=20.0):
    """Annual coupon (%) making the autocall price = par. Bisection on price."""
    def price_at(annual_cpn):
        per_period = annual_cpn / obs_per_year
        return price_autocall(per_period, S0, r, q, model, p, T, obs_per_year,
                              ac, cb, pb, memory, n_steps, n_paths, seed, notional)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = price_at(mid)
        if abs(pm - notional) < tol:
            return mid
        if pm < notional:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
