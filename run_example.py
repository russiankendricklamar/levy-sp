#!/usr/bin/env python3
"""
run_example.py — end-to-end demonstration of the structured-product codebase.

Prices all four products across all seven models on the calibrated S&P 500
parameters, prints a comparison table, and writes it to prices_out.csv.

Run:  python run_example.py
      python run_example.py --asset AAPL --paths 20000
"""
import argparse, sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import bootstrap as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='^SPX', choices=['^SPX', 'AAPL', 'SPY'],
                    help='which calibrated asset to price on (default ^SPX)')
    ap.add_argument('--paths', type=int, default=40000, help='MC paths for path-dependent products')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ctx = B.load()
    prod = ctx['products']
    asset = args.asset
    S0 = ctx['meta']['assets'][asset]['spot']
    r = ctx['rf']
    q = 0.0
    calib = ctx['calib'][asset]

    print(f"Asset={asset}  S0={S0:.2f}  r={r:.4f}  q={q}")
    print(f"MC paths={args.paths}  seed={args.seed}\n")

    rows = []
    for m in ctx['ALL_MODELS']:
        if m not in calib:
            continue
        p = calib[m]['params']
        # European (FFT, all models)
        note = prod.note_participation(S0, r, q, m, p)
        dig  = prod.digital_call(S0, r, q, m, p)
        # Path-dependent (MC, all except CGMY)
        if m == 'CGMY':
            brc = ac = np.nan
        else:
            brc = prod.fair_coupon_brc(S0, r, q, m, p, n_paths=args.paths, seed=args.seed)
            ac  = prod.fair_coupon_ac( S0, r, q, m, p, n_paths=args.paths, seed=args.seed)
        rows.append(dict(Model=m,
                         Note_participation_pct=round(note, 2),
                         Digital_price=round(dig, 4),
                         BRC_coupon_pct=round(brc, 3) if not np.isnan(brc) else np.nan,
                         Autocall_coupon_pct=round(ac, 3)  if not np.isnan(ac)  else np.nan))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv('prices_out.csv', index=False)
    print("\nWritten -> prices_out.csv")

    # model-risk summary vs Black-Scholes
    bs = df[df.Model == 'BS'].iloc[0]
    print("\nModel risk vs Black-Scholes:")
    for _, row in df.iterrows():
        if row.Model == 'BS':
            continue
        gap = ""
        if not np.isnan(row.BRC_coupon_pct):
            gap = f"BRC {100*(row.BRC_coupon_pct/bs.BRC_coupon_pct-1):+.0f}%"
        print(f"  {row.Model:8s} {gap}")


if __name__ == '__main__':
    main()
