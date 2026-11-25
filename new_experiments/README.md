# Revision experiments (ICMIEE 2026 reviewer response)

These scripts produce every number and figure in the revised paper.
Install the pinned versions first. LightGBM and scikit-learn versions change
results slightly, and these are the versions the paper's numbers came from.

    pip install -r new_experiments/requirements.txt

Run from anywhere (the scripts locate `../dataset` themselves), in this order:

| Script | Produces | Reviewer comment |
|---|---|---|
| `e1_reproduce.py` | Table 3 (OOF/test RMSE, NASA score, stacking weights), uncapped-label RMSE | 5 |
| `e2_agree_error.py` | Per-engine agreement vs. absolute error, quartile comparison | 1 |
| `e2b_partial.py` | Partial correlations (predicted RUL, ensemble spread), AUROC | 1 |
| `e3_self_agree.py` | Table 5: seed and bootstrap self-agreement baselines (~15 min) | 2 |
| `e4_prune_cv.py` | Inner-CV feature selection (1-SE rule), repeated latency timing | 3, 4 |
| `figs.py` | Figs. 2 and 3 (saved to `figures/`) | 1, 3 |

`core.py` holds the shared pipeline (same preprocessing, features and
hyperparameters as `code-rul.ipynb`).
