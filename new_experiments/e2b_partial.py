import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import sys, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(rx))]+[rankdata(z) for z in Z])
    res = lambda v: v - Zr @ np.linalg.lstsq(Zr, v, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    return spearmanr(ex, ey)
D = pickle.load(open("e1.pkl","rb"))
for ds in ["FD001","FD003"]:
    t = pd.read_csv(f"{ds}_agree_err.csv"); pred = D[ds]["st"]
    for a in ["agr_top5","agr_rho"]:
        r1 = partial_spearman(t[a], t.abs_err_stack, [pred])
        r2 = partial_spearman(t[a], t.abs_err_stack, [pred, t.pred_spread])
        print(ds, a, "| partial on predRUL: %.3f p=%.3f | on predRUL+spread: %.3f p=%.3f" % (r1[0], r1[1], r2[0], r2[1]))
    r = partial_spearman(t.pred_spread, t.abs_err_stack, [pred]); print(ds, "spread | partial on predRUL: %.3f p=%.3f" % r)
    # AUROC for detecting large errors (top quartile |err|)
    from sklearn.metrics import roc_auc_score
    big = t.abs_err_stack >= t.abs_err_stack.quantile(0.75)
    print(ds, "AUROC big-error: 1-agr_rho %.3f, 1-agr_top5 %.3f, spread %.3f" % (roc_auc_score(big, -t.agr_rho), roc_auc_score(big, -t.agr_top5), roc_auc_score(big, t.pred_spread)))
    print(ds, "rho(agr_rho, predRUL) %.3f" % spearmanr(t.agr_rho, pred)[0])
