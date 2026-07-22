"""Levy model library: characteristic functions + Carr-Madan FFT pricer + BS analytics."""
import numpy as np
from scipy.special import gamma as gammafn
from scipy.stats import norm
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def psi_bs(u, p):
    return -0.5 * p['sigma'] ** 2 * u * u


def psi_merton(u, p):
    s, lam, muJ, dJ = p['sigma'], p['lam'], p['muJ'], p['deltaJ']
    return -0.5 * s * s * u * u + lam * (np.exp(1j * u * muJ - 0.5 * dJ * dJ * u * u) - 1.0)


def psi_vg(u, p):
    s, th, nu = p['sigma'], p['theta'], p['nu']
    return -(1.0 / nu) * np.log(1.0 - 1j * u * th * nu + 0.5 * s * s * nu * u * u)


def psi_nig(u, p):
    a, b, d = p['alpha'], p['beta'], p['delta']
    return -d * (np.sqrt(a * a - (b + 1j * u) ** 2) - np.sqrt(a * a - b * b))


def psi_cgmy(u, p):
    C, G, M, Y = p['C'], p['G'], p['M'], p['Y']
    return C * gammafn(-Y) * ((M - 1j * u) ** Y - M ** Y + (G + 1j * u) ** Y - G ** Y)


PSI = {'BS': psi_bs, 'Merton': psi_merton, 'VG': psi_vg, 'NIG': psi_nig, 'CGMY': psi_cgmy}


def cf_logS(u, t, S0, r, q, model, p):
    psi = PSI[model]
    omega = np.real(-psi(-1j, p))
    return np.exp(1j * u * (np.log(S0) + (r - q + omega) * t) + t * psi(u, p))


def carr_madan(t, S0, r, q, model, p, alpha=1.5, N=2 ** 13, eta=0.25):
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    ku = -b + lam * np.arange(N)
    vj = eta * np.arange(N)
    u = vj - (alpha + 1) * 1j
    cf = cf_logS(u, t, S0, r, q, model, p)
    psi_hat = np.exp(-r * t) * cf / (alpha ** 2 + alpha - vj ** 2 + 1j * (2 * alpha + 1) * vj)
    j = np.arange(N)
    simpson = (3 - (-1) ** j) / 3
    simpson[0] = 1 / 3
    call = np.real(np.exp(-alpha * ku) / np.pi * fft(np.exp(1j * b * vj) * psi_hat * eta * simpson))
    return np.exp(ku), call


def price_call_fft(K_target, t, S0, r, q, model, p, **kw):
    K, C = carr_madan(t, S0, r, q, model, p, **kw)
    m = (K > 0.1 * S0) & (K < 10 * S0) & np.isfinite(C)
    return interp1d(K[m], C[m], kind='cubic')(K_target)


def bs_call(S0, K, r, q, t, sig):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return S0 * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)


def bs_put(S0, K, r, q, t, sig):
    return bs_call(S0, K, r, q, t, sig) - S0 * np.exp(-q * t) + K * np.exp(-r * t)


def bs_iv(price, S0, K, r, q, t, kind='call'):
    intr = (max(S0 * np.exp(-q * t) - K * np.exp(-r * t), 0)
            if kind == 'call' else max(K * np.exp(-r * t) - S0 * np.exp(-q * t), 0))
    if price <= intr + 1e-8:
        return np.nan
    base = bs_call if kind == 'call' else bs_put
    try:
        return brentq(lambda s: base(S0, K, r, q, t, s) - price, 1e-4, 5.0, maxiter=200)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Monte-Carlo path simulator (Levy models only; Heston/Bates -> simulators.py)
# ---------------------------------------------------------------------------
def omega_of(model, p):
    return float(np.real(-PSI[model](-1j, p)))


def _rng_invgauss(mu, lam, n, rng):
    nu = rng.standard_normal(n)
    y = nu * nu
    x = mu + (mu * mu * y) / (2 * lam) - (mu / (2 * lam)) * np.sqrt(4 * mu * lam * y + mu * mu * y * y)
    z = rng.random(n)
    return np.where(z <= mu / (mu + x), x, mu * mu / x)


def simulate_paths(model, p, S0, r, q, T, n_steps, n_paths, rng):
    dt = T / n_steps
    drift = (r - q + omega_of(model, p)) * dt
    logS = np.full(n_paths, np.log(S0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = S0
    for i in range(1, n_steps + 1):
        if model == 'BS':
            inc = p['sigma'] * np.sqrt(dt) * rng.standard_normal(n_paths)
        elif model == 'Merton':
            inc = p['sigma'] * np.sqrt(dt) * rng.standard_normal(n_paths)
            N = rng.poisson(p['lam'] * dt, n_paths)
            inc = inc + np.where(N > 0,
                                  rng.normal(p['muJ'] * N, p['deltaJ'] * np.sqrt(np.maximum(N, 1))), 0.0)
        elif model == 'VG':
            G = rng.gamma(dt / p['nu'], p['nu'], n_paths)
            inc = p['theta'] * G + p['sigma'] * np.sqrt(G) * rng.standard_normal(n_paths)
        elif model == 'NIG':
            a, b, d = p['alpha'], p['beta'], p['delta']
            gam = np.sqrt(a * a - b * b)
            I = _rng_invgauss(d * dt / gam, (d * dt) ** 2, n_paths, rng)
            inc = b * I + np.sqrt(I) * rng.standard_normal(n_paths)
        else:
            raise ValueError(f"simulate_paths: unsupported model '{model}'")
        logS = logS + drift + inc
        out[:, i] = np.exp(logS)
    return out
