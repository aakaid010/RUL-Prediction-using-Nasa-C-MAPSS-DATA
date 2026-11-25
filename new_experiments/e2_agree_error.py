import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
"""Comment 1: does per-engine explanation agreement relate to prediction error?"""
import sys, pickle, itertools, numpy as np, pandas as pd

from core import *
from scipy.stats import spearmanr, mannwhitneyu

D = pickle.load(open("e1.pkl", "rb"))
rows = []
for ds, d in D.items():
    n = len(d["yt"]); P = list(itertools.combinations(NAMES, 2))
    agr_top5 = np.array([np.mean([topk_overlap(np.abs(d["sv"][a][i]), np.abs(d["sv"][b][i])) for a, b in P]) for i in range(n)])
    agr_rho = np.array([np.mean([spearmanr(np.abs(d["sv"][a][i]), np.abs(d["sv"][b][i]))[0] for a, b in P]) for i in range(n)])
    preds = np.column_stack([d["tp"][m] for m in NAMES])
    err_st = np.abs(d["st"] - d["yt"]); err_mean = np.abs(preds.mean(1) - d["yt"])
    spread = preds.std(1)
    res = {"ds": ds}
    for an, av in [("top5", agr_top5), ("rho", agr_rho)]:
        r, p = spearmanr(av, err_st); res[f"rho({an},|err|)"] = (round(r, 3), round(p, 3))
        r, p = spearmanr(av, spread); res[f"rho({an},pred-spread)"] = (round(r, 3), round(p, 3))
        r, p = spearmanr(av, d["yt"]); res[f"rho({an},trueRUL)"] = (round(r, 3), round(p, 3))
    r, p = spearmanr(spread, err_st); res["rho(spread,|err|)"] = (round(r, 3), round(p, 3))
    # flagged vs unflagged: bottom quartile of agreement (none fall below 0.40)
    q = np.quantile(agr_top5, 0.25)
    lo = agr_top5 <= q
    res["n<0.40"] = int((agr_top5 < 0.40).sum())
    res["agr range"] = (round(agr_top5.min(), 2), round(agr_top5.max(), 2), round(agr_top5.mean(), 2))
    res["Q1 thr"] = round(q, 3); res["n_low"] = int(lo.sum())
    res["MAE low/high"] = (round(err_st[lo].mean(), 2), round(err_st[~lo].mean(), 2))
    res["RMSE low/high"] = (round(np.sqrt((err_st[lo] ** 2).mean()), 2), round(np.sqrt((err_st[~lo] ** 2).mean()), 2))
    res["MWU p"] = round(mannwhitneyu(err_st[lo], err_st[~lo]).pvalue, 3)
    # same with rho-based agreement
    q2 = np.quantile(agr_rho, 0.25); lo2 = agr_rho <= q2
    res["rho-agr MAE low/high"] = (round(err_st[lo2].mean(), 2), round(err_st[~lo2].mean(), 2), round(mannwhitneyu(err_st[lo2], err_st[~lo2]).pvalue, 3))
    # the paper's 30-engine subset
    idx = np.sort(np.random.RandomState(SEED).choice(n, 30, replace=False))
    r, p = spearmanr(agr_top5[idx], err_st[idx]); res["30-subset rho(top5,|err|)"] = (round(r, 3), round(p, 3))
    rows.append(res)
    for k, v in res.items(): print(f"{ds} {k}: {v}")
    print()
    pd.DataFrame({"engine": np.arange(1, n + 1), "true_rul": d["yt"], "agr_top5": agr_top5, "agr_rho": agr_rho,
                  "abs_err_stack": err_st, "pred_spread": spread}).to_csv(f"{ds}_agree_err.csv", index=False)
pickle.dump(rows, open("e2.pkl", "wb"))
