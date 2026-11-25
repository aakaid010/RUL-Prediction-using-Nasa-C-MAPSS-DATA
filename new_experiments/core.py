"""Reproduction of the notebook pipeline (code-rul.ipynb) as importable functions."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "dataset"
COLS = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
CAP = 125
SEED = 42
NAMES = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def load(ds, cap_test=True):
    rd = lambda k: pd.read_csv(DATA / f"{k}_{ds}.txt", sep=r"\s+", header=None, names=COLS)
    tr, te = rd("train"), rd("test")
    fr = pd.read_csv(DATA / f"RUL_{ds}.txt", sep=r"\s+", header=None, names=["RUL"])
    tr["RUL"] = np.minimum(tr.groupby("unit_id")["cycle"].transform("max") - tr["cycle"], CAP)
    fr["unit_id"] = range(1, len(fr) + 1)
    last = te.groupby("unit_id")["cycle"].max().rename("last").reset_index().merge(fr, on="unit_id")
    last["tl"] = last["last"] + last["RUL"]
    te = te.merge(last[["unit_id", "tl"]], on="unit_id")
    raw = te["tl"] - te["cycle"]
    te["RUL_uncapped"] = raw
    te["RUL"] = np.minimum(raw, CAP) if cap_test else raw
    te = te.drop(columns="tl")
    sc = [c for c in tr.columns if c not in ("unit_id", "cycle", "RUL")]
    sd = tr[sc].std()
    keep = sd[sd > 1e-5].index.tolist()
    removed = [s for s in sc if s not in keep]
    mn, mx = tr[keep].min(), tr[keep].max()
    rng = (mx - mn).replace(0, 1)
    tr[keep] = (tr[keep] - mn) / rng
    te[keep] = (te[keep] - mn) / rng
    tr = roll(tr[["unit_id", "cycle"] + keep + ["RUL"]], keep)
    te = roll(te[["unit_id", "cycle"] + keep + ["RUL", "RUL_uncapped"]], keep)
    feats = [c for c in tr.columns if c not in ("unit_id", "cycle", "RUL", "RUL_uncapped")]
    return tr, te, feats, keep, removed


def roll(d, sc, w=5):
    d = d.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    g = d.groupby("unit_id")[sc]
    m = g.rolling(w, min_periods=1).mean().reset_index(drop=True)
    m.columns = [f"{s}_rollmean" for s in sc]
    s = g.rolling(w, min_periods=1).std().fillna(0).reset_index(drop=True)
    s.columns = [f"{c}_rollstd" for c in sc]
    return pd.concat([d, m, s], axis=1)


def models(seed=SEED, stochastic=False):
    sub = dict(subsample=0.8, colsample_bytree=0.8) if stochastic else {}
    lsub = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8) if stochastic else {}
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=seed, n_jobs=-1, verbosity=0, **sub),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=seed, n_jobs=-1, verbose=-1, **lsub),
        "CatBoost": CatBoostRegressor(iterations=100, depth=5, learning_rate=0.08, random_state=seed, verbose=0, thread_count=-1, allow_writing_files=False),
    }


def lgbm(seed=SEED):
    return LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=seed, n_jobs=-1, verbose=-1)


def nasa(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def shap_vals(m, X):
    v = shap.TreeExplainer(m).shap_values(X)
    if isinstance(v, list):
        v = v[0]
    return np.asarray(v)


def topk_overlap(a, b, k=5):
    """a, b: importance vectors (higher = more important)."""
    ta = set(np.argsort(-a)[:k]); tb = set(np.argsort(-b)[:k])
    return len(ta & tb) / k
