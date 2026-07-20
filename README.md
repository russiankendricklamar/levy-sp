# Lévy Processes & Stochastic Volatility for Structured Products Pricing

> **Bottom line:** Comparative study of 6 models (BS, Merton, VG, NIG, CGMY, Heston, Bates) for pricing BRC and Phoenix structured products. Calibration via FFT (Carr–Madan) to real option chains (SPX, AAPL, SPY) from Yahoo Finance (2026). Risk analysis: P&L distribution, VaR/CVaR, Greeks.

---

## Overview

This repository contains the code, data, and research behind the paper:

> **"Efficiency of Lévy Processes and Stochastic Volatility Models in Pricing Structured Products: Calibration to Real Option Market Data"** (2026)

The study benchmarks pricing models across three asset classes (SPX index, single stock AAPL, ETF SPY) and two payoff structures (BRC with knock-in barrier, Phoenix autocall with coupon trigger), evaluating model quality both in terms of IV-RMSE fit and downstream P&L/risk metrics.

---

## Models Implemented

| Model | Type | Key Parameters |
|---|---|---|
| Black–Scholes | Diffusion | σ |
| Merton Jump-Diffusion | Jump-diffusion | σ, λ, μ_J, σ_J |
| Variance Gamma (VG) | Pure-jump Lévy | σ, ν, θ |
| Normal Inverse Gaussian (NIG) | Pure-jump Lévy | α, β, δ |
| CGMY | Pure-jump Lévy | C, G, M, Y |
| Heston | Stochastic volatility | κ, θ, ξ, ρ, v₀ |
| Bates | SV + jumps | Heston + Merton jumps |

---

## Methodology

### Calibration
- **Data:** Yahoo Finance option chains (2026), risk-free rate via 13-week T-bill (IRX = 3.71%), dividends included
- **Objective:** Minimize IV-RMSE on mid bid-ask quotes, moneyness K/S₀ ∈ [0.8, 1.2], T ∈ {1, 2} months
- **Optimizer:** Trust Region Reflective (`scipy.optimize`)
- **Pricing kernel:** FFT via Carr–Madan (1999) — see `models/fft_pricer.py`

### Structured Product Pricing
- **BRC (Barrier Reverse Convertible):** Knock-in barrier at 70%, priced via FFT (analytical) and MC path simulation (40,000 paths, CGMY)
- **Phoenix Autocall:** 3 coupon observation dates, coupon trigger 100%, barrier 70%, early redemption trigger 60%
- Monte Carlo engine with QE scheme (Broadie–Kaya, 2008) for Heston/Bates

### Risk Analysis
- P&L distribution: ATM straddle, ΔT = 0.25 yr, DGP = SPX realized
- VaR₉₅, CVaR₉₅, CVaR₉₉ for each model
- Greeks: Delta (Δ), Vega (ν), ATM skew ∂σ/∂(K/S₀)

---

## Repository Structure

```
levy-sv-structured-products/
│
├── data/
│   ├── fetch_chains.py          # Yahoo Finance data loader (yfinance)
│   └── sample/                  # Sample CSVs: SPX, AAPL, SPY
│
├── models/
│   ├── characteristic_functions.py   # φ(u) for all 7 models
│   ├── fft_pricer.py                 # Carr–Madan FFT pricing
│   ├── mc_engine.py                  # Monte Carlo (QE scheme for SV)
│   └── calibration.py                # Calibration loop, IV-RMSE
│
├── products/
│   ├── brc.py                   # BRC pricing (FFT + MC)
│   └── phoenix.py               # Phoenix autocall (MC)
│
├── risk/
│   ├── greeks.py                # Delta, Vega, Skew
│   └── pl_distribution.py       # P&L, VaR, CVaR
│
├── notebooks/
│   ├── 01_calibration.ipynb     # Full calibration workflow
│   ├── 02_structured_pricing.ipynb   # BRC + Phoenix pricing
│   └── 03_risk_analysis.ipynb   # Greeks, P&L, VaR/CVaR
│
├── results/                     # Saved calibration params, tables, plots
│
├── requirements.txt
└── README.md
```

---

## Key Results (Paper Summary)

### Calibration Quality (IV-RMSE, %)

| Model | SPX | AAPL | SPY |
|---|---|---|---|
| Black–Scholes | 6.45 | 4.37 | 6.24 |
| Merton | 2.00 | 2.70 | 1.79 |
| VG | 2.34 | 2.89 | 2.03 |
| NIG | 2.04 | 2.80 | 1.73 |
| **CGMY** | **1.68** | 2.76 | **1.48** |
| **Heston** | 0.91 | **1.13** | 0.81 |
| **Bates** | **0.54** | **0.91** | **0.56** |

### BRC Price (S₀=100, barrier=70%, T=1yr)

| Model | Price | Delta |
|---|---|---|
| BS | 63.50 | 0.539 |
| CGMY | 56.77 | 0.593 |
| Heston | 55.01 | 0.629 |
| Bates | 54.80 | 0.631 |

### P&L Risk (ATM Straddle, DGP=SPX)

| Model | VaR₉₅ | CVaR₉₅ | CVaR₉₉ |
|---|---|---|---|
| BS | 220.8 | 255.3 | 578.4 |
| Merton | 243.2 | 293.5 | 646.0 |
| Heston | 259.2 | 327.7 | 693.4 |
| Bates | 256.7 | 321.6 | 684.6 |

---

## Limitations & Caveats

1. **Data:** Only Yahoo Finance (2026) — limited expiry grid, no OTC surfaces
2. **Calibration:** 2 maturities only; longer-dated smile not captured
3. **CGMY Y parameter:** Near Y→2 triggers numerical instability in FFT
4. **QE scheme:** Valid for Feller condition violations but introduces discretization bias
5. **FX extension:** EUR/USD tested but omitted from main results (illiquid chain)
6. **Rough volatility** (rBergomi, rough Heston) — not benchmarked; likely better fit for SPX short-dated skew

---

## Quickstart

```bash
git clone https://github.com/russiankendricklamar/levy-sv-structured-products.git
cd levy-sv-structured-products
pip install -r requirements.txt
jupyter notebook notebooks/01_calibration.ipynb
```

---

## Requirements

```
numpy>=1.26
scipy>=1.12
pandas>=2.2
yfinance>=0.2
matplotlib>=3.8
jupyter
```

---

## Citation

```bibtex
@misc{galkin2026levy,
  title     = {Efficiency of L{\'{e}}vy Processes and Stochastic Volatility Models
               in Pricing Structured Products: Calibration to Real Option Market Data},
  author    = {Galkin, Egor},
  year      = {2026},
  note      = {Working paper}
}
```

---

## License

MIT
