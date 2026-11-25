import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
"""Reproduce Table 2/3 and dump everything needed for later analyses."""
import sys, pickle, numpy as np

from core import *

out = {}
for ds in ["FD001", "FD003"]:
    tr, te, F, keep, removed = load(ds)
    print(ds, "removed:", removed, "kept:", len(keep), "features:", len(F), "train rows", len(tr), "test rows", len(te))
    X, y, g = tr[F], tr["RUL"], tr["unit_id"]
    li = te.groupby("unit_id")["cycle"].idxmax()
    Xt, yt, ytu = te.loc[li, F].reset_index(drop=True), te.loc[li, "RUL"].values, te.loc[li, "RUL_uncapped"].values
    oof = {n: np.zeros(len(X)) for n in NAMES}; tp = {}; fm = {}
    for n in NAMES:
        for a, b in GroupKFold(3).split(X, y, g):
            m = models()[n]; m.fit(X.iloc[a], y.iloc[a]); oof[n][b] = m.predict(X.iloc[b])
        m = models()[n]; m.fit(X, y); fm[n] = m; tp[n] = m.predict(Xt)
        print(f"  {n:14s} OOF {rmse(y, oof[n]):.2f} test {rmse(yt, tp[n]):.2f} NASA {nasa(yt, tp[n]):.2f}  | uncapped-label RMSE {rmse(ytu, tp[n]):.2f}")
    meta = Ridge(alpha=1.0, positive=True).fit(np.column_stack([oof[n] for n in NAMES]), y)
    st = meta.predict(np.column_stack([tp[n] for n in NAMES]))
    print(f"  Stack test {rmse(yt, st):.2f} NASA {nasa(yt, st):.2f} uncapped {rmse(ytu, st):.2f} w={np.round(meta.coef_, 3)}")
    print(f"  #test engines with true RUL>125: {(ytu > 125).sum()}")
    sv = {n: shap_vals(fm[n], Xt) for n in NAMES}
    out[ds] = dict(F=F, Xt=Xt, yt=yt, ytu=ytu, tp=tp, st=st, sv=sv, keep=keep, removed=removed)
pickle.dump(out, open("e1.pkl", "wb"))
