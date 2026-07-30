# RUL-Prediction-using-Nasa-C-MAPSS-DATA

This project predicts the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. It covers the full pipeline from raw sensor data to a stacked ensemble model, along with SHAP-based explainability and feature pruning analysis.

## Dataset

The [NASA C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) contains run-to-failure simulation data for a fleet of turbofan engines. Each engine starts with an unknown initial wear state and degrades over time until failure. This project uses subsets **FD001** and **FD003**.

Each row of data corresponds to one engine at one operational cycle, with the following columns:

- `unit` - engine ID
- `cycle` - operational cycle number
- `op1`, `op2`, `op3` - operational settings
- `s1` to `s21` - sensor measurements

## File Structure

```
RUL/
├── CMAPSSData/
│   ├── Damage Propagation Modeling.pdf   # original NASA paper describing the dataset
│   ├── readme.txt                        # NASA dataset documentation
│   ├── train_FD001.txt ... train_FD004.txt
│   ├── test_FD001.txt  ... test_FD004.txt
│   └── RUL_FD001.txt   ... RUL_FD004.txt
├── CMAPSSData.zip                        # zipped version of the above
├── code.ipynb                            # main notebook (data prep, modeling, SHAP, pruning)
└── README.md
```

## Pipeline Overview

The notebook is organized into four stages:

### 1. Data Preprocessing
- Load raw `train_*.txt` / `test_*.txt` / `RUL_*.txt` files for FD001 and FD003
- Compute RUL labels for training data (max cycle - current cycle)
- Compute RUL labels for test data using the provided end-of-life RUL values
- Apply a piecewise-linear RUL cap (125 cycles) to reflect that engines degrade negligibly early in life
- Drop constant / near-constant sensor columns that carry no information
- Min-max normalize all remaining features (scaler fit on training data only, to avoid leakage)

### 2. Feature Engineering
- Rolling window statistics (mean and standard deviation) over a 5-cycle window, computed per engine, for every sensor/operational feature

### 3. Model Training
Four base regressors are trained per subset (FD001, FD003) using Group K-Fold cross-validation (grouped by engine unit, so no engine appears in both train and validation folds):

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

Out-of-fold predictions from the base models are combined into a stacking ensemble with a Ridge regression meta-learner. Models are evaluated with:

- RMSE (root mean squared error)
- NASA/PHM08 scoring function (asymmetric penalty that penalizes late predictions more than early ones, since predicting a longer remaining life than actual is riskier in maintenance settings)

Trained models are saved as `.joblib` files.

### 4. Explainability and Feature Pruning
- SHAP (TreeExplainer) values are computed for each base model
- Cross-model explanation agreement is measured using top-k feature overlap and rank agreement, to check whether different models agree on which sensors matter
- A trust flag is generated based on an agreement threshold
- Leakage-free SHAP-based feature importance ranking (computed via cross-validation) is used to test accuracy vs. inference-time tradeoffs as the feature count is reduced (5, 10, 15, 20, 25, 30... features)

## Models Used

| Component | Method |
|---|---|
| Base learners | Random Forest, XGBoost, LightGBM, CatBoost |
| Ensemble | Stacking with Ridge regression meta-learner |
| Cross-validation | Group K-Fold (3 folds, grouped by engine unit) |
| Explainability | SHAP TreeExplainer |
| Evaluation metrics | RMSE, NASA/PHM08 asymmetric scoring function |

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
shap
scipy
matplotlib
joblib
```

## Running the Notebook

This notebook was converted from Google Colab and is set up to run on Kaggle:

1. Add `CMAPSSData.zip` as a Kaggle dataset input (or place it in the working directory)
2. Run all cells top to bottom
3. Processed data, trained models, results, and figures are written to the working directory as the pipeline progresses

## Reference

Saxena, A. and Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
