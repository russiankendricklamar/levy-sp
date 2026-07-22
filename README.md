# Lévy Processes and Stochastic-Volatility Models for Structured-Product Pricing

Reproducible research code for pricing structured products under seven models: Black-Scholes, Merton jump-diffusion, Variance Gamma, NIG, CGMY, Heston, and Bates.

This repository accompanies the master's thesis **"The Effectiveness of Lévy Processes and Stochastic-Volatility Models in Pricing Structured Products"** and provides a unified implementation of characteristic-function pricing, Carr–Madan FFT, Monte Carlo path simulation, calibrated market parameters, and benchmark result tables.

---

## Scope

The project studies how model choice affects:

- implied-volatility smile fit
- pricing of structured products
- Greeks
- delta-hedging error distributions

**Products covered**

- Capital-protected note
- ATM digital call
- Barrier reverse convertible (BRC)
- Phoenix autocall

**Models covered**

| Class | Models |
|---|---|
| Diffusion | Black-Scholes |
| Exponential Lévy | Merton, Variance Gamma, NIG, CGMY |
| Stochastic volatility | Heston, Bates |

---

## Data and assumptions

Market inputs were collected from live option chains in July 2026 for three underlyings: the S&P 500 index (`^SPX`), Apple (`AAPL`), and the SPY ETF.

Key assumptions baked into the calibration artifacts:

- Risk-free rate: 3.71%
- Assets: `^SPX`, `AAPL`, `SPY`, each with 8 option maturities
- Spots: SPX 7457.69, AAPL 333.74, SPY 743.29
- European products priced via Carr–Madan FFT; path-dependent products priced via Monte Carlo
- CGMY is European-only in this implementation — no path simulator

---

## Main results

Calibration quality (IV-RMSE, %) is strongest for Bates across all three assets, followed by Heston, then pure Lévy models, with Black-Scholes weakest:

| Model | S&P 500 | AAPL | SPY |
|---|---:|---:|---:|
| BS | 6.45 | 4.37 | 6.24 |
| Merton | 2.00 | 2.70 | 1.79 |
| VG | 2.34 | 2.89 | 2.03 |
| NIG | 2.04 | 2.80 | 1.73 |
| CGMY | 1.68 | 2.76 | 1.48 |
| Heston | 0.91 | 1.13 | 0.81 |
| Bates | 0.54 | 0.91 | 0.56 |

On SPX-calibrated parameters, model choice materially shifts fair product terms:

| Model | Note participation % | Digital price | BRC coupon % | Autocall coupon % |
|---|---:|---:|---:|---:|
| BS | 63.50 | 5.3929 | 4.279 | 4.482 |
| Merton | 58.45 | 5.9031 | 6.118 | 5.561 |
| VG | 58.58 | 5.6905 | 5.835 | 5.673 |
| NIG | 58.27 | 5.7617 | 5.880 | 5.563 |
| CGMY | 56.77 | 5.9336 | — | — |
| Heston | 55.01 | 6.2922 | 7.772 | 6.572 |
| Bates | 54.80 | 6.3124 | 7.635 | 6.782 |

Hedging-error tables show strongly negative skew and heavy kurtosis across all hedger models, consistent with unhedgeable jump/gap risk dominating the left tail of P&L rather than model choice itself.

---

## Repository structure

```text
levy-sp/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_example.py
├── lib/
│   ├── bootstrap.py
│   ├── levylib.py
│   ├── svmodels.py
│   ├── simulators.py
│   └── products.py
├── data/
│   ├── real_smiles.parquet
│   ├── calibration_real.json
│   ├── data_meta.json
│   ├── calibration_rmse_table.csv
│   ├── master_prices_real.csv
│   ├── greeks_table.csv
│   └── hedging_errors_table.csv
└── thesis.pdf
```

---

## Installation

```bash
git clone https://github.com/russiankendricklamar/levy-sp.git
cd levy-sp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24`, `scipy>=1.10`, `pandas>=1.5`, `pyarrow>=10.0`. Pure CPU — no GPU required.

---

## Quick start

```bash
python run_example.py                  # all products, all models, S&P 500
python run_example.py --asset AAPL     # same on Apple
python run_example.py --paths 100000   # higher Monte Carlo precision
```

Output: a console comparison table plus `prices_out.csv`.

---

## Usage

```python
import sys
sys.path.insert(0, "lib")

import bootstrap as B
ctx = B.load()

S0 = ctx["meta"]["assets"]["^SPX"]["spot"]
r = ctx["rf"]
q = 0.0
prod = ctx["products"]
params = ctx["calib"]["^SPX"]["Bates"]["params"]

note     = prod.note_participation(S0, r, q, "Bates", params)
digital  = prod.digital_call(S0, r, q, "Bates", params)
brc      = prod.fair_coupon_brc(S0, r, q, "Bates", params, n_paths=40000, seed=0)
autocall = prod.fair_coupon_ac(S0, r, q, "Bates", params, n_paths=40000, seed=0)
```

All pricers accept any of the seven model strings: `"BS"`, `"Merton"`, `"VG"`, `"NIG"`, `"CGMY"`, `"Heston"`, `"Bates"`.

> ⚠️ CGMY supports European products only (no path simulator).

### Direct option pricing

```python
sv = ctx["svmodels"]
sv.price_call_any(K=5000, T=1.0, S0=S0, r=r, q=0.0, model="Heston", p=params)
```

### Product customisation

| Product | Key arguments (defaults) |
|---|---|
| Capital-protected note | `T=3.0`, `notional=100` |
| Digital call | `T=1.0`, `payout=10` |
| BRC | `T=1.0`, `ki=0.70`, `n_paths=40000`, `seed=0` |
| Phoenix Autocall | `T=3.0`, `obs_per_year=4`, `ac=1.00`, `cb=0.70`, `pb=0.60`, `memory=True` |

---

## Methodology

Under exponential Lévy dynamics, the discounted asset price is martingale-corrected via

S_t = S_0 · exp((r − q + ω)t + X_t),  where ω = −ψ(−i)

and ψ(u) is the characteristic exponent of the underlying Lévy process.

European options are priced with the Carr–Madan damped Fourier transform, using default numerical settings α = 1.5, N = 2¹³, η = 0.25. Heston and Bates use closed-form characteristic functions; path-dependent products are simulated by Monte Carlo, with Heston/Bates paths generated via Andersen's Quadratic-Exponential (QE) scheme, which stays stable even when the Feller condition is violated by calibration.

---

## Reproducibility

- European products (note, digital) are **deterministic** given the FFT grid — bit-for-bit reproducible.
- Path-dependent products (BRC, autocall) are Monte Carlo estimates; results match reference values within ±0.05–0.3 pp at 40,000 paths. Increase `n_paths` for tighter convergence.

---

## Limitations

This is a research implementation, not a production pricing library.

- Calibration reflects a single market snapshot (July 2026), not a repeated time series
- CGMY has no path simulator in the current implementation
- FX smile calibration is excluded — listed FX-ETF options were too illiquid
- No transaction costs, liquidity effects, slippage, or funding asymmetry
- Single-underlying products only — no local-stochastic-volatility, rough-volatility, or stochastic-rate extensions

---

## Thesis

The full thesis document is included as `thesis.pdf`.

---

## Citation

```bibtex
@mastersthesis{galkin2026levy,
  author = {Galkin, Egor S.},
  title  = {The Effectiveness of Lévy Processes and Stochastic-Volatility Models
            in Pricing Structured Products},
  year   = {2026},
  month  = {July},
  url    = {https://github.com/russiankendricklamar/levy-sp}
}
```

---

## License

MIT — see `LICENSE`.
