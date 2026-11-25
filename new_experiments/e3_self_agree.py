import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
"""Comment 2: calibrate cross-model agreement against self-agreement of the same learner."""
import sys, pickle, itertools, numpy as np, pandas as pd

from core import *
from scipy.stats import spearmanr

def glob_metrics(Ia, Ib, k=5):
    ra, rb = np.argsort(-Ia), np.argsort(-Ib)
    top = len(set(ra[:k]) & set(rb[:k])) / k
    exact = np.mean(ra[:k] == rb[:k])
    pa, pb = np.argsort(ra), np.argsort(rb)  # rank position of each feature
    n = len(Ia); agree = tot = 0
    for i, j in itertools.combinations(range(n), 2):
        agree += (pa[i] < pa[j]) == (pb[i] < pb[j]); tot += 1
    return dict(top5=top, exact5=exact, pairwise=agree / tot, spearman=spearmanr(pa, pb)[0])

def inst_rho(Sa, Sb):
    return np.mean([spearmanr(np.abs(Sa[i]), np.abs(Sb[i]))[0] for i in range(len(Sa))])

def summarize(svs):
    """svs: list of SHAP matrices; returns mean metrics over all pairs."""
    rows = []
    for a, b in itertools.combinations(range(len(svs)), 2):
        m = glob_metrics(np.abs(svs[a]).mean(0), np.abs(svs[b]).mean(0)); m["inst_rho"] = inst_rho(svs[a], svs[b]); rows.append(m)
    return pd.DataFrame(rows).mean()

R = 5
out = []
for ds in ["FD001", "FD003"]:
    tr, te, F, *_ = load(ds)
    X, y, g = tr[F], tr["RUL"], tr["unit_id"]
    li = te.groupby("unit_id")["cycle"].idxmax(); Xt = te.loc[li, F].reset_index(drop=True)
    # collinearity diagnostics
    C = X.corr().abs().values; iu = np.triu_indices(len(F), 1)
    print(ds, "feature pairs |r|>0.9:", int((C[iu] > 0.9).sum()), "of", len(iu[0]))
    rm = [f for f in F if f.endswith("_rollmean")]
    Cm = X[rm].corr().abs().values; ium = np.triu_indices(len(rm), 1)
    print(ds, "median |r| among rollmean features:", round(float(np.median(Cm[ium])), 3))
    # cross-model (default configs, seed 42)
    base = {}
    for n in NAMES:
        m = models()[n]; m.fit(X, y); base[n] = shap_vals(m, Xt)
    cross = summarize([base[n] for n in NAMES]); cross.name = "cross-model"
    print(ds, "CROSS", cross.round(3).to_dict())
    out.append(dict(ds=ds, kind="cross-model", learner="all 6 pairs", **cross.to_dict()))
    engines = np.unique(g)
    for n in NAMES:
        # (a) seed perturbation
        if n in ("XGBoost", "LightGBM"):
            svs = []
            for s in range(R):
                m = models(seed=s, stochastic=True)[n]; m.fit(X, y); svs.append(shap_vals(m, Xt))
            kind = "seed (subsample/colsample=0.8)"
        else:
            svs = []
            for s in range(R):
                m = models(seed=s)[n]; m.fit(X, y); svs.append(shap_vals(m, Xt))
            kind = "seed"
        r = summarize(svs); print(ds, n, kind, r.round(3).to_dict())
        out.append(dict(ds=ds, kind=kind, learner=n, **r.to_dict()))
        # (b) engine bootstrap, default config
        svs = []
        for s in range(R):
            rs = np.random.RandomState(100 + s); pick = rs.choice(engines, len(engines), replace=True)
            idx = np.concatenate([np.where(g.values == e)[0] for e in pick])
            m = models()[n]; m.fit(X.iloc[idx], y.iloc[idx]); svs.append(shap_vals(m, Xt))
        r = summarize(svs); print(ds, n, "engine-bootstrap", r.round(3).to_dict())
        out.append(dict(ds=ds, kind="engine-bootstrap", learner=n, **r.to_dict()))
pd.DataFrame(out).to_csv("e3_self_agreement.csv", index=False)
