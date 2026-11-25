import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral"],"mathtext.fontset":"stix","font.size":8,
  "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#666","axes.linewidth":0.6,
  "xtick.color":"#444","ytick.color":"#444","axes.labelcolor":"#222","legend.frameon":False})
B,O="#2a78d6","#eb6834"; out="figures/"; os.makedirs(out, exist_ok=True)
# Fig: agreement vs error
fig,ax=plt.subplots(1,2,figsize=(3.4,1.85),sharey=True)
for a,ds in zip(ax,["FD001","FD003"]):
    t=pd.read_csv(f"{ds}_agree_err.csv"); r,p=spearmanr(t.agr_rho,t.abs_err_stack)
    q=t.agr_rho.quantile(.25)
    a.axvspan(t.agr_rho.min()-.02,q,color="#eee",lw=0,zorder=0)
    a.scatter(t.agr_rho,t.abs_err_stack,s=10,color=B,edgecolor="white",linewidth=0.4,zorder=2)
    a.set_title(f"{ds}  ($\\rho_s$ = {r:.2f}, p < 0.001)",fontsize=8); a.set_xlabel("Per-engine agreement $\\bar\\rho$")
    a.grid(axis="y",color="#e5e5e5",lw=0.5); a.tick_params(length=2)
ax[0].set_ylabel("|Error| (cycles)"); ax[0].text(0.03,0.95,"bottom\nquartile",transform=ax[0].transAxes,va="top",fontsize=7,color="#555")
fig.tight_layout(pad=0.3); fig.savefig(out+"agreement_vs_error.pdf"); fig.savefig(out+"agreement_vs_error.png",dpi=200)
# Fig: pruning
d=pd.read_csv("e4_prune_cv.csv")
fig,ax=plt.subplots(1,2,figsize=(3.4,1.85))
for a,ds in zip(ax,["FD001","FD003"]):
    s=d[d.ds==ds]; x=np.arange(len(s)); lab=[str(k) if k<=30 else "All" for k in s.k]
    a.errorbar(x,s.cv_rmse,yerr=s.cv_se,color=B,marker="o",ms=3.5,lw=1.4,capsize=1.5,label="Inner-CV RMSE")
    a.plot(x,s.test_rmse,color=O,marker="s",ms=3.5,lw=1.4,ls="--",label="Test RMSE")
    i=list(s.k).index(10); a.scatter([x[i]],[s.cv_rmse.iloc[i]],s=70,facecolor="none",edgecolor="#222",lw=0.9,zorder=5,label="1-SE choice ($k$=10)")
    a.set_xticks(x); a.set_xticklabels(lab); a.set_title(ds,fontsize=8); a.set_xlabel("Features retained, $k$")
    a.grid(axis="y",color="#e5e5e5",lw=0.5); a.tick_params(length=2)
ax[0].set_ylabel("RMSE (cycles)")
h,l=ax[0].get_legend_handles_labels(); o=[l.index(n) for n in ["Inner-CV RMSE","Test RMSE","1-SE choice ($k$=10)"]]; fig.legend([h[j] for j in o],[l[j] for j in o],loc="upper center",ncol=3,fontsize=7,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(pad=0.3,rect=(0,0,1,0.9)); fig.savefig(out+"pruning_cv.pdf"); fig.savefig(out+"pruning_cv.png",dpi=200)
