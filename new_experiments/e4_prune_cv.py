import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
"""Comments 3 & 4: choose k by inner GroupKFold CV (test set untouched), report test once;
and measure inference latency properly (repeated timing)."""
import sys, time, numpy as np, pandas as pd

from core import *
from sklearn.model_selection import GroupKFold

KS = [5, 10, 15, 20, 25, 30]

def shap_rank(X, y, g, folds=3, sample=100, seed=SEED):
    imps = []
    for f, (a, b) in enumerate(GroupKFold(folds).split(X, y, g), 1):
        m = lgbm(); m.fit(X.iloc[a], y.iloc[a])
        rs = np.random.RandomState(seed + f); s = rs.choice(len(b), min(sample, len(b)), replace=False)
        imps.append(np.abs(shap_vals(m, X.iloc[b].iloc[s])).mean(0))
    return list(X.columns[np.argsort(-np.mean(imps, 0))])

def trunc_eval_idx(df, rs, per_engine=5):
    """sample random truncation points per engine to mimic last-cycle test protocol"""
    idx = []
    for _, grp in df.groupby("unit_id"):
        idx.extend(rs.choice(grp.index.values, min(per_engine, len(grp)), replace=False))
    return np.array(idx)

out = []; lat = []
for ds in ["FD001", "FD003"]:
    tr, te, F, *_ = load(ds)
    X, y, g = tr[F], tr["RUL"], tr["unit_id"]
    li = te.groupby("unit_id")["cycle"].idxmax(); Xt_all = te.loc[li, F]; yt = te.loc[li, "RUL"].values
    Ks = KS + [len(F)]
    cv = {k: [] for k in Ks}; cvt = {k: [] for k in Ks}
    # outer GroupKFold on TRAINING engines only; ranking recomputed inside each fold's train part
    for fo, (a, b) in enumerate(GroupKFold(5).split(X, y, g)):
        Xa, ya, ga = X.iloc[a], y.iloc[a], g.iloc[a]
        rank = shap_rank(Xa.reset_index(drop=True), ya.reset_index(drop=True), ga.reset_index(drop=True))
        tidx = trunc_eval_idx(tr.iloc[b], np.random.RandomState(fo))
        for k in Ks:
            m = lgbm(); m.fit(Xa[rank[:k]], ya)
            cv[k].append(rmse(y.iloc[b], m.predict(X.iloc[b][rank[:k]])))
            cvt[k].append(rmse(y.loc[tidx], m.predict(X.loc[tidx, rank[:k]])))
    cvm = {k: np.mean(v) for k, v in cv.items()}; cvs = {k: np.std(v, ddof=1) / np.sqrt(len(v)) for k, v in cv.items()}
    k_min = min(cvm, key=cvm.get)
    k_1se = min(k for k in Ks if cvm[k] <= cvm[k_min] + cvs[k_min])
    # final ranking on all training data (as in paper), evaluate ONLY the selected k on test
    rank = shap_rank(X, y, g)
    for k in Ks:
        m = lgbm(); m.fit(X[rank[:k]], y)
        Xt = Xt_all[rank[:k]]
        p = m.predict(Xt)
        # latency: median of repeated timings, batch of 100 engines and single engine
        Xn = Xt.values.astype(np.float64); one = Xn[:1]
        bst = m.booster_
        for _ in range(20): bst.predict(Xn)
        tb = []; ts = []
        for _ in range(300):
            t0 = time.perf_counter(); bst.predict(Xn); tb.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); bst.predict(one); ts.append(time.perf_counter() - t0)
        out.append(dict(ds=ds, k=k, cv_rmse=cvm[k], cv_se=cvs[k], cv_trunc_rmse=np.mean(cvt[k]),
                        test_rmse=rmse(yt, p), test_nasa=nasa(yt, p),
                        selected=("min" if k == k_min else "") + ("/1SE" if k == k_1se else ""),
                        batch100_ms_median=1000 * np.median(tb), batch100_ms_iqr=1000 * (np.quantile(tb, .75) - np.quantile(tb, .25)),
                        single_ms_median=1000 * np.median(ts), single_ms_iqr=1000 * (np.quantile(ts, .75) - np.quantile(ts, .25))))
    print(ds, "k_min", k_min, "k_1se", k_1se, "top-10 rank:", rank[:10])
df = pd.DataFrame(out); df.to_csv("e4_prune_cv.csv", index=False)
pd.set_option("display.width", 250); print(df.round(4).to_string())
