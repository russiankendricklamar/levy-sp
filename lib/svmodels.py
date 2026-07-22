"""Stochastic-volatility models: Heston + Bates characteristic functions,
unified Carr-Madan FFT pricer over Levy + SV models."""
import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import levylib


def cf_heston(u, t, S0, r, q, p):
    kappa, theta, sigma, rho, v0 = p['kappa'], p['theta'], p['sigma'], p['rho'], p['v0']
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi ** 2 + sigma ** 2 * (1j * u + u ** 2))
    g = (xi - d) / (xi + d)
    e = np.exp(-d * t)
    C = ((r - q) * 1j * u * t
         + kappa * theta / sigma ** 2 * ((xi - d) * t - 2 * np.log((1 - g * e) / (1 - g))))
    D = (xi - d) / sigma ** 2 * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v0 + 1j * u * np.log(S0))


def cf_bates(u, t, S0, r, q, p):
    kappa, theta, sigma, rho, v0 = p['kappa'], p['theta'], p['sigma'], p['rho'], p['v0']
    lam, muJ, delJ = p['lam'], p['muJ'], p['deltaJ']
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi ** 2 + sigma ** 2 * (1j * u + u ** 2))
    g = (xi - d) / (xi + d)
    e = np.exp(-d * t)
    kbar = np.exp(muJ + 0.5 * delJ ** 2) - 1
    jump = lam * t * (np.exp(1j * u * muJ - 0.5 * delJ ** 2 * u ** 2) - 1)
    drift = (r - q - lam * kbar) * 1j * u * t
    C = (drift
         + kappa * theta / sigma ** 2 * ((xi - d) * t - 2 * np.log((1 - g * e) / (1 - g))))
    D = (xi - d) / sigma ** 2 * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v0 + 1j * u * np.log(S0) + jump)


SV_CF = {'Heston': cf_heston, 'Bates': cf_bates}


def cf_any(u, t, S0, r, q, model, p):
    if model in SV_CF:
        return SV_CF[model](u, t, S0, r, q, p)
    return levylib.cf_logS(u, t, S0, r, q, model, p)


def carr_madan_any(t, S0, r, q, model, p, alpha=1.5, N=2 ** 13, eta=0.25):
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    ku = -b + lam * np.arange(N)
    vj = eta * np.arange(N)
    u = vj - (alpha + 1) * 1j
    cf = cf_any(u, t, S0, r, q, model, p)
    psi_hat = np.exp(-r * t) * cf / (alpha ** 2 + alpha - vj ** 2 + 1j * (2 * alpha + 1) * vj)
    j = np.arange(N)
    simp = (3 - (-1) ** j) / 3
    simp[0] = 1 / 3
    call = np.real(np.exp(-alpha * ku) / np.pi * fft(np.exp(1j * b * vj) * psi_hat * eta * simp))
    return np.exp(ku), call


def price_call_any(Kt, t, S0, r, q, model, p, **kw):
    K, C = carr_madan_any(t, S0, r, q, model, p, **kw)
    m = (K > 0.05 * S0) & (K < 20 * S0) & np.isfinite(C)
    return interp1d(K[m], C[m], kind='cubic')(Kt)


ALL_MODELS = ['BS', 'Merton', 'VG', 'NIG', 'CGMY', 'Heston', 'Bates']
