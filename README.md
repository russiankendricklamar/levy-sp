# Lévy Processes and Stochastic-Volatility Models in Structured-Product Pricing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible codebase for the master's thesis:
> **"The Effectiveness of Lévy Processes and Stochastic-Volatility Models in Pricing Structured Products"**  
> Egor S. Galkin, July 2026

Seven option-pricing models (Black–Scholes, Merton, VG, NIG, CGMY, Heston, Bates) are calibrated to **real market implied-volatility smiles** (Yahoo Finance, July 2026) across three asset classes (S&P 500, AAPL, SPY) and applied to price four structured products.

---

## Key Results

| Finding | Detail |
|---|---|
| Best smile fit | Bates: IV-RMSE 0.5–0.9% vs 4–6% for Black–Scholes |
| Model risk in prices | Fair BRC coupon shifts up to **+82%** (Heston/Bates vs BS) on real data |
| Hedging | Unhedgeable jump/gap risk dominates P&L tail — richer model does not remove it |

---

## Repository Layout

```
levy-sp/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_example.py          # end-to-end demo: prices all products × all models
├── lib/
│   ├── bootstrap.py        # loader: B.load() → single context dict
│   ├── levylib.py          # Lévy char. functions, Carr–Madan FFT, BS analytics
│   ├── svmodels.py         # Heston + Bates char. functions, unified FFT dispatcher
│   ├── simulators.py       # MC path simulation (exact Lévy schemes + Andersen QE)
│   └── products.py         # pricers: capital-protected note, digital, BRC, autocall
└── data/
    ├── real_smiles.parquet         # cleaned market smiles (SPX / AAPL / SPY)
    ├── calibration_real.json       # calibrated model parameters
    ├── data_meta.json              # spots, risk-free rate, metadata
    ├── calibration_rmse_table.csv  # fit quality (IV-RMSE per model/asset)
    ├── master_prices_real.csv      # reference product prices (for reproducibility check)
    ├── greeks_table.csv
    └── hedging_errors_table.csv    # P&L VaR / CVaR of delta-hedging errors
```

---

## Quick Start

```bash
git clone https://github.com/russiankendricklamar/levy-sp.git
cd levy-sp
pip install -r requirements.txt

python run_example.py                  # prices all products on S&P 500
python run_example.py --asset AAPL     # same on Apple
python run_example.py --paths 100000   # higher MC precision
```

Output: console table + `prices_out.csv`.

---

## Usage

### Load everything

```python
import sys; sys.path.insert(0, "lib")
import bootstrap as B

ctx = B.load()
S0     = ctx["meta"]["assets"]["^SPX"]["spot"]          # 7457.69
r      = ctx["rf"]                                       # 0.0371
prod   = ctx["products"]
params = ctx["calib"]["^SPX"]["Bates"]["params"]         # calibrated Bates params
```

### Price structured products

```python
# Capital-protected note (3y) — fair participation rate (%)
prod.note_participation(S0, r, 0.0, "Bates", params)

# ATM digital call (1y, payout=10) — price
prod.digital_call(S0, r, 0.0, "Bates", params)

# Barrier Reverse Convertible (1y, KI=70%) — fair coupon (%)
prod.fair_coupon_brc(S0, r, 0.0, "Bates", params, n_paths=40_000, seed=0)

# Phoenix Autocall (3y, quarterly, AC=100/CB=70/PB=60, memory) — annual coupon (%)
prod.fair_coupon_ac(S0, r, 0.0, "Bates", params, n_paths=40_000, seed=0)
```

All pricers accept any of the 7 model strings: `"BS"`, `"Merton"`, `"VG"`, `"NIG"`, `"CGMY"`, `"Heston"`, `"Bates"`.  
> ⚠️ CGMY supports **European products only** (no path simulator).

### Direct option pricing (FFT)

```python
sv = ctx["svmodels"]
sv.price_call_any(K=5000, T=1.0, S0=S0, r=r, q=0.0, model="Heston", p=params)
```

### Custom parameters / own calibration

All pricers accept an arbitrary parameter dict `p`. See `data/calibration_real.json` for the key schema of each model.

---

## Product Customisation

| Product | Key arguments (defaults) |
|---|---|
| Capital-protected note | `T=3.0`, `notional=100` |
| Digital call | `T=1.0`, `payout=10` |
| BRC | `T=1.0`, `ki=0.70`, `n_paths=40000`, `seed=0` |
| Phoenix Autocall | `T=3.0`, `obs_per_year=4`, `ac=1.00`, `cb=0.70`, `pb=0.60`, `memory=True` |

Example — BRC with 60% barrier, 2-year tenor:

```python
prod.fair_coupon_brc(S0, r, 0.0, "Heston", params, T=2.0, ki=0.60)
```

---

## Models

| Model | Type | Characteristic exponent ψ(u) |
|---|---|---|
| Black–Scholes | GBM | −½σ²u² |
| Merton | Jump-diffusion | −½σ²u² + λ(e^{iuμ_J − ½δ²_J u²} − 1) |
| Variance Gamma | Pure-jump Lévy | −(1/ν) ln(1 − iuθν + ½σ²νu²) |
| NIG | Pure-jump Lévy | −δ(√(α²−(β+iu)²) − √(α²−β²)) |
| CGMY | Pure-jump Lévy | CΓ(−Y)[(M−iu)^Y − M^Y + (G+iu)^Y − G^Y] |
| Heston | Stochastic vol | CIR variance process, closed-form CF |
| Bates | SV + jumps | Heston + Merton jumps |

FFT pricing follows Carr & Madan (1999) with damping parameter α = 1.5, N = 2¹³ grid points, η = 0.25 step, Simpson quadrature weights. Verified against analytic BS to max error 2.2 × 10⁻⁷.

Path simulation: exact incremental schemes for BS/Merton/VG/NIG; Andersen Quadratic-Exponential (QE) scheme for Heston/Bates (stable under Feller condition violation); CGMY — European FFT only.

---

## Reproducibility

Reference prices are stored in `data/master_prices_real.csv`.  
- **European** products (note, digital): exact FFT — deterministic, bit-for-bit reproducible.  
- **Path-dependent** (BRC, autocall): Monte Carlo, matches reference within ±0.05–0.3 pp at `n_paths=40000`; increase paths for tighter tolerance.

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
pandas>=1.5
pyarrow>=10.0
```

Pure CPU — no GPU required.

---

## Citation

If you use this codebase, please cite:

```bibtex
@mastersthesis{galkin2026levy,
  author  = {Galkin, Egor S.},
  title   = {The Effectiveness of Lévy Processes and Stochastic-Volatility Models
             in Pricing Structured Products},
  school  = {},
  year    = {2026},
  month   = {July},
  url     = {https://github.com/russiankendricklamar/levy-sp}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
