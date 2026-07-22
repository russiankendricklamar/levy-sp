"""
bootstrap.py — load all models + calibrated params + market data from the
local codebase (no external dependencies beyond numpy/scipy/pandas/pyarrow).

Usage:
    import bootstrap as B
    ctx = B.load()                              # dict with everything
    ctx['ALL_MODELS']                           # ['BS','Merton','VG','NIG','CGMY','Heston','Bates']
    ctx['calib']['^SPX']['Bates']['params']     # calibrated Bates params for SPX
    ctx['rf']                                   # risk-free rate (0.0371)

Directory layout expected (relative to this file):
    lib/   levylib.py  svmodels.py  simulators.py  products.py  bootstrap.py
    data/  real_smiles.parquet  calibration_real.json  data_meta.json  ...
"""
import os
import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DATA = os.path.join(_ROOT, 'data')

# make lib modules importable by plain name
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def load():
    import levylib
    import svmodels
    import simulators
    import products

    calib  = json.load(open(os.path.join(_DATA, 'calibration_real.json')))
    smiles = pd.read_parquet(os.path.join(_DATA, 'real_smiles.parquet'))
    meta   = json.load(open(os.path.join(_DATA, 'data_meta.json')))

    return dict(
        levylib=levylib,
        svmodels=svmodels,
        simulators=simulators,
        products=products,
        calib=calib,
        smiles=smiles,
        meta=meta,
        rf=meta['risk_free'],
        ALL_MODELS=svmodels.ALL_MODELS,
    )
