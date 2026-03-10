# %% [markdown]
# # Massive CV Ensemble
# * 5 folds x 40 seeds x 5 Configs GBSA Model
# * Use GBSA and LGBM

# %%
import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    brier_score,
)
import sys
import platform
import sklearn
import sksurv
import kagglehub
import os
from Utils.Config import Config, GBSAConfig, TrialResult
from dataclasses import asdict

from Utils.utils import (
    set_seed,
    save_cv_result_json,
    KFold_val,
    create_features,
    save_config_yaml,
    HORIZONS
)

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from Utils.Config import GBSAConfig
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
warnings.filterwarnings("ignore")

# %% [markdown]
# # Enviorment report

# %%

print()
print("Python: ", sys.version.split()[0])
print("OS: ", platform.platform())
print("Scikit-learn: ", sklearn.__version__)
print("Scikit-survival: ", sksurv.__version__)

# %% [markdown]
# # Paths

# %%
COMP_DIR = kagglehub.competition_download('WiDSWorldWide_GlobalDathon26')
metadata_path = os.path.join(COMP_DIR, 'metaData.csv')
train_path = os.path.join(COMP_DIR, 'train.csv')
test_path = os.path.join(COMP_DIR, 'test.csv')
seed = 42
set_seed(seed)

train_df = pd.read_csv(train_path)
train_processed = create_features(train_df)

print("Metadata path: ", metadata_path)
print("Train path: ", train_path)
print("Test path: ", test_path)
print()

# %%

def sample_gbsa_config(trial: Trial, seed: int = 42) -> GBSAConfig:
    config = GBSAConfig(
        # core boosting
        loss="coxph",
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),

        # tree structure
        max_depth=trial.suggest_int("max_depth", 1, 6),
        max_features=trial.suggest_categorical(
            "max_features",
            [None, "sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        ),
        max_leaf_nodes=trial.suggest_categorical(
            "max_leaf_nodes",
            [None, 8, 16, 31, 63]
        ),

        # split control
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.0, 0.05),

        # tree criterion
        criterion="friedman_mse",

        # regularization
        ccp_alpha=trial.suggest_float("ccp_alpha", 1e-8, 1e-1, log=True),
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5),

        # early stopping
        validation_fraction=trial.suggest_float("validation_fraction", 0.1, 0.3),
        n_iter_no_change=trial.suggest_categorical("n_iter_no_change", [None, 5, 10, 20]),
        tol=trial.suggest_float("tol", 1e-5, 1e-2, log=True),

        # misc
        random_state=seed,
        warm_start=False,
        verbose=0,
    )
    return config

#GradientBoostingSurvivalAnalysis(**asdict(config.gbsa_config))
def objective(trial: Trial) -> float:
    config = Config(
        seed=seed,
        gbsa_config=sample_gbsa_config(trial, seed),
        cv_n_splits=5,
        cv_n_repeats=10
    )
    
    dir = os.path.join("Trials", "GBSA", f"trial_{trial.number}")
    os.makedirs(dir, exist_ok=True)
    config_file_path = os.path.join(dir, "config.yaml")
    cv_resutl_file_path = os.path.join(dir, "cv_results.json")
    save_config_yaml(config, config_file_path)
    
    model = GradientBoostingSurvivalAnalysis(**asdict(config.gbsa_config))
    start = time.perf_counter()
    cv_results= KFold_val(model, train_processed, seed, n_splits=config.cv_n_splits, n_repeats=config.cv_n_repeats, verbose=False)
    elapsed = time.perf_counter() - start
    
    trial.set_user_attr(
        "kfold_result",
        asdict(cv_results)
    )
    
    print(f"\nTrial {trial.number} completed in {elapsed:.2f} seconds with")
    print(f"hybrid score {cv_results.hybrid_score:.4f} and \nC-index {cv_results.c_index:.4f} and \nmean Brier {cv_results.mean_brier:.4f}")
    print(f"hybrid STD {cv_results.std_hybrid:.4f} and \nC-index STD {cv_results.std_c_index:.4f} and \nmean Brier STD {cv_results.std_mean_brier:.4f}")
    
    trial_result = TrialResult(
        trial_id=trial.number,
        config=config,
        result=cv_results
    )
    save_cv_result_json(trial_result, cv_resutl_file_path)
    
    return cv_results.hybrid_score

storage = JournalStorage(
    JournalFileBackend("gbsa_journal.log")
)

study = optuna.create_study(
    study_name="gbsa_survival",
    storage=storage,
    load_if_exists=True,
    direction="maximize",
    sampler=TPESampler(seed=seed, multivariate=True),
)

study.optimize(objective, n_trials=300)



# %%
