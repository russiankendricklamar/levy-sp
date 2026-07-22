"""Unified Monte-Carlo path simulator for Levy + stochastic-vol models.
All models share martingale-corrected risk-neutral drift; paths returned as price arrays."""
import numpy as np
import levylib


def _omega(model, p):
    return float(np.real(-levylib.PSI[model](-1j, p)))


def _rng_ig(mu, lam, n, rng):
    nu = rng.standard_normal(n)
    y = nu * nu
    x = mu + (mu * mu * y) / (2 * lam) - (mu / (2 * lam)) * np.sqrt(4 * mu * lam * y + mu * mu * y * y)
    z = rng.random(n)
    return np.where(z <= mu / (mu + x), x, mu * mu / x)


def simulate(model, p, S0, r, q, T, n_steps, n_paths, rng):
    """Dispatch to appropriate scheme for Levy or SV models."""
    if model in ('Heston', 'Bates'):
        return simulate_qe(model, p, S0, r, q, T, n_steps, n_paths, rng)
    # Levy models
    dt = T / n_steps
    drift = (r - q + _omega(model, p)) * dt
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
            I = _rng_ig(d * dt / gam, (d * dt) ** 2, n_paths, rng)
            inc = b * I + np.sqrt(I) * rng.standard_normal(n_paths)
        elif model == 'CGMY':
            raise NotImplementedError('CGMY path simulation not supported; use FFT for European products')
        else:
            raise ValueError(f"Unknown model: {model}")
        logS = logS + drift + inc
        out[:, i] = np.exp(logS)
    return out


def simulate_qe(model, p, S0, r, q, T, n_steps, n_paths, rng):
    """Andersen Quadratic-Exponential scheme for Heston/Bates.
    Stable under Feller condition violation, typical for calibrated params."""
    kappa, theta, sig, rho, v0 = p['kappa'], p['theta'], p['sigma'], p['rho'], p['v0']
    dt = T / n_steps
    E = np.exp(-kappa * dt)
    psic = 1.5  # switching threshold
    # log-price constants (Andersen martingale-correct central discretization)
    K0 = -rho * kappa * theta / sig * dt
    K1 = 0.5 * dt * (kappa * rho / sig - 0.5) - rho / sig
    K2 = 0.5 * dt * (kappa * rho / sig - 0.5) + rho / sig
    K3 = 0.5 * dt * (1 - rho * rho)
    K4 = K3
    has_j = (model == 'Bates')
    if has_j:
        lam, muJ, delJ = p['lam'], p['muJ'], p['deltaJ']
        kbar = np.exp(muJ + 0.5 * delJ ** 2) - 1
    lnS = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, v0)
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = S0
    for i in range(1, n_steps + 1):
        m = theta + (v - theta) * E
        s2 = (v * sig * sig * E * (1 - E) / kappa
              + theta * sig * sig * (1 - E) ** 2 / (2 * kappa))
        psi = s2 / (m * m)
        vnext = np.empty(n_paths)
        # psi <= psic: quadratic branch
        idx = psi <= psic
        if idx.any():
            pin = psi[idx]
            b2 = 2 / pin - 1 + np.sqrt(2 / pin) * np.sqrt(2 / pin - 1)
            b2 = np.maximum(b2, 0)
            a = m[idx] / (1 + b2)
            Zv = rng.standard_normal(idx.sum())
            vnext[idx] = a * (np.sqrt(b2) + Zv) ** 2
        # psi > psic: exponential branch
        jdx = ~idx
        if jdx.any():
            pin = psi[jdx]
            pp = (pin - 1) / (pin + 1)
            beta = (1 - pp) / m[jdx]
            U = rng.random(jdx.sum())
            vnext[jdx] = np.where(U <= pp, 0.0, np.log((1 - pp) / (1 - U)) / beta)
        Zx = rng.standard_normal(n_paths)
        drift = (r - q) * dt
        if has_j:
            drift = drift - lam * kbar * dt
        lnS = (lnS + drift + K0 + K1 * v + K2 * vnext
               + np.sqrt(np.maximum(K3 * v + K4 * vnext, 0)) * Zx)
        if has_j:
            N = rng.poisson(lam * dt, n_paths)
            lnS = lnS + np.where(N > 0,
                                   rng.normal(muJ * N, delJ * np.sqrt(np.maximum(N, 1))), 0.0)
        v = vnext
        out[:, i] = np.exp(lnS)
    return out
