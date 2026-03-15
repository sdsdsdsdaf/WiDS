from typing import Any

from catboost import CatBoostRegressor
from sksurv.linear_model import CoxnetSurvivalAnalysis
import numpy
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sksurv.metrics import concordance_index_censored
import xgboost as xgb
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
import torch
import torch.nn as nn

import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchtuples")


import numpy as np


class StepSurvivalFunction:
    def __init__(self, times, surv_values):
        self.x = np.asarray(times, dtype=float)
        self.y = np.asarray(surv_values, dtype=float)
        self.domain = (float(self.x[0]), float(self.x[-1]))

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.x, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.y) - 1)
        return self.y[idx]


def breslow_baseline(event, time, risk_score, already_exp=False, clip_value=20.0):
    """
    Estimate baseline cumulative hazard H0(t) and baseline survival S0(t)
    using the Breslow estimator.

    Parameters
    ----------
    already_exp : bool
        If True, risk_score is already hazard ratio scale.
        If False, risk_score is log-risk and exp() will be applied.
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float).reshape(-1)

    order = np.argsort(time)
    time = time[order]
    event = event[order]
    risk_score = risk_score[order]

    if already_exp:
        exp_risk = np.clip(risk_score, 1e-12, 1e12)
    else:
        exp_risk = np.exp(np.clip(risk_score, -clip_value, clip_value))

    unique_event_times = np.unique(time[event])

    cumulative_hazard = 0.0
    baseline_cumhaz = []

    for t in unique_event_times:
        d_i = np.sum((time == t) & event)
        risk_set_sum = np.sum(exp_risk[time >= t])

        h_i = 0.0 if risk_set_sum <= 0 else d_i / risk_set_sum
        cumulative_hazard += h_i
        baseline_cumhaz.append(cumulative_hazard)

    baseline_cumhaz = np.asarray(baseline_cumhaz, dtype=float)
    baseline_surv = np.exp(-baseline_cumhaz)

    return unique_event_times, baseline_cumhaz, baseline_surv


def make_cox_survival_functions(risk, event_times, baseline_cumhaz, already_exp=False, clip_value=20.0):
    """
    Build individual survival functions from Cox risk scores.

    Parameters
    ----------
    already_exp : bool
        If True, risk is already hazard ratio scale.
        If False, risk is log-risk and exp() will be applied.
    """
    risk = np.asarray(risk, dtype=float).reshape(-1)
    event_times = np.asarray(event_times, dtype=float)
    baseline_cumhaz = np.asarray(baseline_cumhaz, dtype=float)

    surv_fns = []
    for r in risk:
        if already_exp:
            hr = np.clip(r, 1e-12, 1e12)
        else:
            hr = np.exp(np.clip(r, -clip_value, clip_value))

        surv = np.exp(-baseline_cumhaz * hr)
        surv_fns.append(StepSurvivalFunction(event_times, surv))

    return surv_fns

class CoxnetWithStandardScaler(BaseEstimator):
    def __init__(
        self,
        use_scaler: bool = True,
        fit_baseline_model: bool = True,
        n_alphas: int = 100,
        alphas: Any | None = None,
        alpha_min_ratio: str | float = "auto",
        l1_ratio: float = 0.5,
        penalty_factor: Any | None = None,
        normalize: bool = False,
        copy_X: bool = True,
        tol: float = 1e-7,
        max_iter: int = 100000,
        verbose: bool = False,
    ):
        self.use_scaler = use_scaler
        self.fit_baseline_model = fit_baseline_model
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.scaler_ = None
        self.model_ = None

    def fit(self, X:NDArray, y:NDArray):
        X_cpy = X.copy()

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            X_cpy = self.scaler_.fit_transform(X_cpy)

        self.model_ = CoxnetSurvivalAnalysis(
            n_alphas=self.n_alphas,
            alphas=self.alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            l1_ratio=self.l1_ratio,
            penalty_factor=self.penalty_factor,
            normalize=self.normalize,
            copy_X=self.copy_X,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            fit_baseline_model=self.fit_baseline_model,
        )
        
        self.model_.fit(X_cpy, y)
        return self

    def predict(self, X:NDArray):
        X_cpy = X.copy()
        if self.scaler_ is not None:
            X_cpy = self.scaler_.transform(X_cpy)
        return self.model_.predict(X_cpy)

    def predict_survival_function(self, X:NDArray):
        X_cpy = X.copy()
        if self.scaler_ is not None:
            X_cpy = self.scaler_.transform(X_cpy)
        return self.model_.predict_survival_function(X_cpy)
    

class DeepHit(BaseEstimator):

    def __init__(
        self,
        num_durations=50,
        hidden_dims=(128,64),
        batch_norm=True,
        dropout=0.1,
        alpha=0.2,
        sigma=0.1,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=0.0,
        batch_size=256,
        epochs=100,
        patience=10,
        verbose=False
    ):
        self.num_durations = num_durations
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.alpha = alpha
        self.sigma = sigma
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.scaler = None


    def _get_optimizer(self):

        name = self.optimizer.lower()

        if name == "adam":
            return tt.optim.Adam
        if name == "adamw":
            return tt.optim.AdamW
        if name == "sgd":
            return tt.optim.SGD

        raise ValueError(f"Unknown optimizer: {self.optimizer}")


    def fit(self, X, y, X_val=None, y_val=None):

        X = np.asarray(X, dtype=np.float32)
        
        if self.scaler is None:
            self.scaler = StandardScaler()

        X_train = self.scaler.fit_transform(X)
        durations = y["time"].astype(np.float32)
        events = y["event"].astype(np.int64)

        self.labtrans = LabTransDiscreteTime(self.num_durations)
        y_train = self.labtrans.fit_transform(durations, events)
        

        net = tt.practical.MLPVanilla(
            in_features=X_train.shape[1],
            num_nodes=list(self.hidden_dims),
            out_features=self.labtrans.out_features,
            batch_norm=self.batch_norm,
            dropout=self.dropout
        )


        optimizer_cls = self._get_optimizer()

        self.model = DeepHitSingle(
            net,
            optimizer_cls,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.labtrans.cuts
        )

        self.model.optimizer.set_lr(self.lr)

        try:
            self.model.optimizer.set("weight_decay", self.weight_decay)
        except:
            pass

        callbacks = []
        if self.patience:
            callbacks.append(tt.callbacks.EarlyStopping(patience=self.patience))

        val_data = None

        if X_val is not None and y_val is not None:

            dur_val = y_val["time"].astype(np.float32)
            ev_val = y_val["event"].astype(np.int64)
            X_val = np.asarray(X_val, dtype=np.float32)
            X_val_norm = self.scaler.transform(X_val)
            val_data = (
                X_val_norm  ,
                self.labtrans.transform(dur_val, ev_val)
            )

        self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            val_data=val_data,
            verbose=self.verbose
        )

        self.time_bins = np.asarray(self.labtrans.cuts)

        return self


    def predict(self, X):
        """
        risk score (for C-index)
        """
        X = np.asarray(X, dtype=np.float32)
        X_norm = X
        if self.scaler is not None:
            X_norm = self.scaler.transform(X)
        else:
            print("Scaler is not exist")

        pmf = self.model.predict_pmf(X_norm)
        
        assert pmf.shape[1] == len(self.time_bins)
        exp_time = (pmf * self.time_bins.reshape(1,-1)).sum(axis=1)

        return exp_time


    def predict_survival_function(self, X):

        X = np.asarray(X, dtype=np.float32)
        X_norm = self.scaler.transform(X)   

        surv_df = self.model.predict_surv_df(X_norm)

        times = surv_df.index.values
        surv = surv_df.values.T

        fns = []

        for s in surv:

            def fn(t, times=times, s=s):

                idx = np.searchsorted(times, t, side="right") - 1
                idx = np.clip(idx,0,len(times)-1)

                return s[idx]

            fn.domain = (times.min(), times.max())

            fns.append(fn)

        return fns
    
class DeepSurv(BaseEstimator):
    def __init__(
        self,
        hidden_dims=(128, 64),
        batch_norm=True,
        dropout=0.1,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=0.0,
        batch_size=256,
        epochs=100,
        patience=10,
        verbose=False,
    ):
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.scaler = None

    def _get_optimizer(self):
        name = self.optimizer.lower()
        if name == "adam":
            return tt.optim.Adam
        if name == "adamw":
            return tt.optim.AdamW
        if name == "sgd":
            return tt.optim.SGD
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)

        if self.scaler is None:
            self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X)

        durations = y["time"].astype(np.float32)
        events = y["event"].astype(np.float32)

        net = tt.practical.MLPVanilla(
            in_features=X_train.shape[1],
            num_nodes=list(self.hidden_dims),
            out_features=1,           # CoxPH는 출력 1개
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            output_bias=False,
        )

        optimizer_cls = self._get_optimizer()
        self.model = CoxPH(net, optimizer_cls)

        self.model.optimizer.set_lr(self.lr)
        try:
            self.model.optimizer.set("weight_decay", self.weight_decay)
        except:
            pass

        callbacks = []
        if self.patience:
            callbacks.append(tt.callbacks.EarlyStopping(patience=self.patience))

        val_data = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            X_val = self.scaler.transform(X_val)
            dur_val = y_val["time"].astype(np.float32)
            ev_val = y_val["event"].astype(np.float32)
            val_data = (X_val, (dur_val, ev_val))

        self.log = self.model.fit(
            X_train,
            (durations, events),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            val_data=val_data,
            verbose=self.verbose,
        )

        # baseline hazard / survival 추정
        self.model.compute_baseline_hazards()
        return self

    def predict(self, X):
        """
        Risk score for C-index.
        Higher = higher risk.
        """
        X = np.asarray(X, dtype=np.float32)
        X = self.scaler.transform(X) if self.scaler is not None else X

        # CoxPH/DeepSurv의 예측은 log-risk 계열 점수라고 보면 됨
        risk = self.model.predict(X).reshape(-1)
        return -risk

    def predict_survival_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self.scaler.transform(X) if self.scaler is not None else X

        surv_df = self.model.predict_surv_df(X)
        times = surv_df.index.values.astype(float)
        surv = surv_df.values.T

        fns = []
        for s in surv:
            def fn(t, times=times, s=s):
                idx = np.searchsorted(times, t, side="right") - 1
                idx = np.clip(idx, 0, len(times) - 1)
                return s[idx]

            fn.domain = (float(times.min()), float(times.max()))
            fns.append(fn)

        return fns
    
class XGBCoxWrapper(BaseEstimator):
    def __init__(
        self,
        eta=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        num_boost_round=500,
        tree_method="hist",
        random_state=42,
        verbosity=False,
        already_exp=False,
    ):
        self.eta = eta
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.num_boost_round = num_boost_round
        self.tree_method = tree_method
        self.random_state = random_state
        self.verbosity = verbosity
        self.already_exp = already_exp
        
        self.event_times_ = None
        self.baseline_surv = None
        self.baseline_cumhaz = None

    @staticmethod
    def _check_y(y):
        if getattr(y, "dtype", None) is None or y.dtype.names is None:
            raise ValueError("y must be a structured array with fields ('event', 'time').")
        if "event" not in y.dtype.names or "time" not in y.dtype.names:
            raise ValueError("y must contain fields 'event' and 'time'.")

        event = np.asarray(y["event"], dtype=bool)
        time = np.asarray(y["time"], dtype=float)

        if np.any(time <= 0):
            raise ValueError("All survival times must be positive.")

        return event, time

    def fit(self, X, y):
        event, time = self._check_y(y)

        params = {
            "objective": "survival:cox",
            "eta": self.eta,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "tree_method": self.tree_method,
            "seed": self.random_state,
            "verbosity": self.verbosity,
        }

        dtrain = xgb.DMatrix(X, label=time)
        self.model_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
        )

        train_risk = self.model_.predict(dtrain)

        self.event_times_, self.baseline_cumhaz, self.baseline_surv_ = breslow_baseline(
            event=event,
            time=time,
            risk_score=train_risk,
            already_exp=self.already_exp
        )

        return self

    def predict(self, X):
        """
        Return log-risk score.
        Higher value = higher risk.
        """
        dtest = xgb.DMatrix(X)
        risk = self.model_.predict(dtest)
        return np.asarray(risk, dtype=float).reshape(-1)

    def predict_survival_function(self, X):
        risk = self.predict(X)
        return make_cox_survival_functions(
            risk=risk,
            event_times=self.event_times_,
            baseline_cumhaz=self.baseline_cumhaz,
            already_exp=self.already_exp
        )


class CatBoostCoxWrapper(BaseEstimator):
    def __init__(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        min_data_in_leaf=1,
        random_strength=1.0,
        bagging_temperature=0.0,
        rsm=1.0,
        loss_function="Cox",
        eval_metric="Cox",
        random_state=42,
        verbose=False,
        already_exp=False
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.rsm = rsm
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.verbose = verbose
        self.already_exp = already_exp
        
        self.event_times_ = None
        self.baseline_surv = None
        self.baseline_cumhaz = None

    @staticmethod
    def _check_y(y):
        if getattr(y, "dtype", None) is None or y.dtype.names is None:
            raise ValueError("y must be a structured array with fields ('event', 'time').")
        if "event" not in y.dtype.names or "time" not in y.dtype.names:
            raise ValueError("y must contain fields 'event' and 'time'.")

        event = np.asarray(y["event"], dtype=bool)
        time = np.asarray(y["time"], dtype=float)

        if np.any(time <= 0):
            raise ValueError("All survival times must be positive.")

        return event, time

    @staticmethod
    def _encode_catboost_cox_label(event, time):
        """
        CatBoost Cox label encoding:
        - event observed  -> +time
        - censored        -> -time
        """
        return np.where(event, time, -time).astype(float)

    def fit(self, X, y):
        event, time = self._check_y(y)
        label = self._encode_catboost_cox_label(event, time)

        self.model_ = CatBoostRegressor(
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            rsm=self.rsm,
            random_seed=self.random_state,
            verbose=self.verbose,
        )

        self.model_.fit(X, label, verbose=self.verbose)

        train_risk = self.model_.predict(X)

        self.event_times_, self.baseline_cumhaz ,self.baseline_surv_ = breslow_baseline(
            event=event,
            time=time,
            risk_score=train_risk,
            already_exp=self.already_exp
        )

        return self

    def predict(self, X):
        """
        Return log-risk score.
        Higher value = higher risk.
        """
        risk = self.model_.predict(X)
        return np.asarray(risk, dtype=float).reshape(-1)

    def predict_survival_function(self, X):
        risk = self.predict(X)
        return make_cox_survival_functions(
            risk=risk,
            event_times=self.event_times_,
            baseline_cumhaz=self.baseline_cumhaz,
            already_exp=self.already_exp
        )


if __name__ == "__main__":
    
    import kagglehub
    import os
    import pandas as pd
    from utils import create_features, make_surv_y, HORIZONS, compute_hybrid_score, get_surv_pred_from_model,set_seed
    from sklearn.model_selection import train_test_split
    
    set_seed(42)
    
    COMP_DIR = kagglehub.competition_download('WiDSWorldWide_GlobalDathon26')
    metadata_path = os.path.join(COMP_DIR, 'metaData.csv')
    train_path = os.path.join(COMP_DIR, 'train.csv')
    test_path = os.path.join(COMP_DIR, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_processed = create_features(train_df)
    test_processed = create_features(test_df)

    y_full = make_surv_y(
        event=train_processed['event'],
        time=train_processed['time_to_hit_hours']
    )
    
    X_full = train_processed.drop(columns=['event', 'time_to_hit_hours'])
    X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, test_size=0.33, random_state=42)

    event_horizon = np.array(HORIZONS).copy()
    event_horizon[-1] = min(event_horizon[-1], y_valid['time'].max() - 1e-6)

    print("Event horizons:", event_horizon)

    
    model = CatBoostCoxWrapper(random_state=42, already_exp=False)
    model.fit(X_train, y_train)
    
    risk_score = model.predict(X_valid)
    print("train risk min/max:", risk_score.min(), risk_score.max())
    pred_surv = get_surv_pred_from_model(model, X_valid, event_horizon)
    
    #Valid
    result = compute_hybrid_score(y_train, y_valid, risk_score, pred_surv, event_horizon)
    print(result)
  

        
    
    