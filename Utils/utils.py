import json

import numpy as np
from sklearn.base import BaseEstimator, clone
import pandas as pd
import warnings
import time
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    brier_score,
)
from tqdm.auto import tqdm
import yaml
import os
from collections import Counter

try:
    from Utils.Config import Config, MetricOuput, KFoldResult, TrialResult, EnsembleModel
except ImportError:
    from Config import Config, MetricOuput, KFoldResult, TrialResult, EnsembleModel
from dataclasses import asdict
from dataclasses import is_dataclass, fields
from typing import Type, TypeVar
import optuna
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

HORIZONS= np.array([12.0, 24.0, 48.0, 72.0])  # 예측할 시간 간격 (시간 단위)

T = TypeVar("T")

def build_gbsa_model_from_trial(trial: optuna.Trial, seed=42) -> GradientBoostingSurvivalAnalysis:
    params = trial.params.copy()
    params["random_state"] = seed
    model = GradientBoostingSurvivalAnalysis(**params)
    return model

def load_top_trial_oofs(
    trials: list[optuna.Trial],
    dir:str = None,
)-> dict[int, dict]:
    result = {}
    trial_sort_by_value = sorted(trials, key= lambda x: x.value, reverse=True)
    trial_numbers = [t.number for t in trial_sort_by_value]
    
    for num in trial_numbers:
        
        trial_result = {}
        meta_file_path = os.path.join(dir, f"trial_{num}", "meta.json")
        oof_hit_file_path = os.path.join(dir, f"trial_{num}","oof_hit.npy")
        oof_risk_file_path = os.path.join(dir, f"trial_{num}","oof_risk.npy")
        oof_surv_file_path = os.path.join(dir, f"trial_{num}","oof_surv.npy")
        meta = {}
        
        with open(meta_file_path, "r") as f:
            meta = json.load(f)
        trial_result['value'] = meta['value']
        trial_result['value_std'] = meta['user_attrs']['kfold_result']['std_hybrid']
        trial_result['trial_id'] = meta['trial_id']
        trial_result['oof_risk'] = np.load(oof_risk_file_path)
        trial_result['oof_surv'] = np.load(oof_surv_file_path)
        trial_result['oof_hit'] = np.load(oof_hit_file_path)
        
        result[num] = trial_result
        
    return result

def save_top_trial_oofs(
    trial: optuna.Trial,
    data: pd.DataFrame,
    horizons: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 4,
    trial_dir:str = None,
):
    model = build_gbsa_model_from_trial(trial, seed)
    oof_result = make_oof_predictions(
        model=model,
        data=data,
        horizons=horizons,
        seed=seed,
        n_splits=n_splits,
        n_repeats=n_repeats,
        verbose=False,
    )
    oof_result_surv = oof_result["final_oof_surv"]
    oof_result_hit = 1 - oof_result_surv
    oof_result_risk = oof_result["final_oof_risk"]
    
    np.save(os.path.join(trial_dir, "oof_surv.npy"), oof_result_surv)
    np.save(os.path.join(trial_dir, "oof_hit.npy"), oof_result_hit)
    np.save(os.path.join(trial_dir, "oof_risk.npy"), oof_result_risk)

    meta = {
        "trial_id": trial.number,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }
    
    with open(os.path.join(trial_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def is_exist_pred(trial_dir:str, file_list=["oof_surv.npy", "oof_hit.npy", "oof_risk.npy", "meta.json"]):
    for file_name in file_list:
        if not os.path.exists(os.path.join(trial_dir, file_name)):
            return False
    
    return True

def get_top_trial_oofs(
    study: optuna.Study,
    data: pd.DataFrame,
    horizons: np.ndarray,
    out_dir: str=None,
    top_ratio: float = 0.3,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 4,
) -> dict[int, dict]:

    trials:list[optuna.Trial] = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials = sorted(trials, key=lambda x: x.value, reverse=True)
    top_k = int(len(trials) * top_ratio)
    print(f"Saving OOF predictions for top {top_k} trials to {out_dir}...")
    print("Best trial value:", trials[0].value)
    print("Best Tral Number:", trials[0].number)
    
    top_trials = trials[:top_k]
    for one_trial in top_trials:
        trial_dir = os.path.join(out_dir, f"trial_{one_trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        if not is_exist_pred(trial_dir):
            save_top_trial_oofs(
                trial=one_trial,
                data=data,
                horizons=horizons,
                seed=seed,
                n_splits=n_splits,
                n_repeats=n_repeats,
                trial_dir=trial_dir
            )
    
    
    return load_top_trial_oofs(top_trials, dir=out_dir)
        
        
def from_dict(dataclass_type: Type[T], data: dict) -> T:
    """Recursively convert dict to dataclass."""
    
    kwargs = {}

    for f in fields(dataclass_type):
        value = data.get(f.name)

        if value is None:
            kwargs[f.name] = None
            continue

        field_type = f.type

        # Nested dataclass
        if is_dataclass(field_type):
            kwargs[f.name] = from_dict(field_type, value)
        else:
            kwargs[f.name] = value

    return dataclass_type(**kwargs)


def set_seed(seed=42):
    np.random.seed(seed)
    
def make_surv_y(event, time):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float))
    
def get_eval_horizons(y_train, y_valid, horizons=HORIZONS, eps=1e-6):
    upper = min(y_train["time"].max(), y_valid["time"].max()) - eps
    eval_horizons = np.asarray(horizons, dtype=float).copy()
    eval_horizons[eval_horizons >= upper] = upper
    return eval_horizons
    
def save_config_yaml(config: Config, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

def save_cv_result_json(result: TrialResult, path: str):
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
        
def load_config_yaml(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    return from_dict(Config, data)

def load_cv_result_json(path: str) -> TrialResult:
    with open(path) as f:
        data = json.load(f)

    return from_dict(TrialResult, data)
    
def get_surv_pred_from_model(model, X, horizons=None):
    """
    Returns survival probabilities S(t) = P(T > t)
    shape: (n_samples, n_horizons)
    """
    surv_fns = model.predict_survival_function(X)
    pred_surv = np.empty((len(surv_fns), len(horizons)), dtype=float)
    
    for i, fn in enumerate(surv_fns):
        # Avoid evaluating outside function domain
        t_min, t_max = fn.domain
        eval_times = np.clip(horizons, t_min, t_max)
        pred_surv[i, :] = fn(eval_times)

    return pred_surv

def get_hit_pred_from_model(model, X, horizons=None):
    """
    Returns hit probabilities P(T <= t)
    shape: (n_samples, n_horizons)
    """
    pred_surv = get_surv_pred_from_model(model, X, horizons)
    return 1.0 - pred_surv

def compute_brier_scores(y_train, y_valid, pred_surv, horizons=None):
    """
    pred_surv must be survival probabilities S(t), not hit probabilities.
    """
    times, scores = brier_score(
        y_train,
        y_valid,
        pred_surv,
        horizons
    )
    return times, scores

def compute_c_index(y_true:Surv, risk_score):
    """
    y_true: structured array from Surv.from_arrays(...)
    risk_score: higher means riskier / earlier event
    """
    return concordance_index_censored(
        y_true["event"],
        y_true["time"],
        risk_score
    )[0]
    
def compute_mean_brier(y_train, y_valid, pred_surv, horizons=None):
    _, scores = compute_brier_scores(y_train, y_valid, pred_surv, horizons)
    return float(np.mean(scores))


def compute_hybrid_score(y_train, y_valid, risk_score, pred_surv, horizons=None) -> MetricOuput:
    """
    Example hybrid:
      0.3 * C-index + 0.7 * (1 - mean Brier)

    If your competition uses a different normalization/aggregation,
    adjust this formula accordingly.
    """
    cidx = compute_c_index(y_valid, risk_score)
    mean_bs = compute_mean_brier(y_train, y_valid, pred_surv, horizons)
    hybrid = 0.3 * cidx + 0.7 * (1.0 - mean_bs)

    return MetricOuput(c_index=cidx, mean_brier=mean_bs, hybrid_score=hybrid)

# 가능하다면  Std도 구하게 수정 예정 -> 2차원 배열을 활용하여
def make_oof_predictions(
    model: BaseEstimator,
    data: pd.DataFrame,
    horizons: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 3,
    verbose: bool = False,
):
    kfold = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    n_samples = len(data)
    n_horizons = len(horizons)

    # Final aggregated OOF across repeats
    final_oof_risk_sum = np.zeros(n_samples, dtype=float)
    final_oof_surv_sum = np.zeros((n_samples, n_horizons), dtype=float)
    final_oof_count = np.zeros(n_samples, dtype=int)

    # Per-repeat independent OOF storage
    repeat_oof_results = []

    # Current repeat accumulators
    repeat_oof_risk = np.zeros(n_samples, dtype=float)
    repeat_oof_surv = np.zeros((n_samples, n_horizons), dtype=float)
    repeat_oof_count = np.zeros(n_samples, dtype=int)

    for fold_idx, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(data)),
        total=n_splits * n_repeats,
        desc="KFold OOF",
        leave=False,
    ):
        repeat_idx = fold_idx // n_splits
        fold_in_repeat = fold_idx % n_splits

        # Safety: reset per-repeat arrays at the first fold of each repeat
        if fold_in_repeat == 0:
            repeat_oof_risk.fill(0.0)
            repeat_oof_surv.fill(0.0)
            repeat_oof_count.fill(0)

        X_train = data.iloc[train_idx].drop(columns=["time_to_hit_hours", "event"])
        y_train = make_surv_y(
            event=data.iloc[train_idx]["event"],
            time=data.iloc[train_idx]["time_to_hit_hours"],
        )

        X_val = data.iloc[val_idx].drop(columns=["time_to_hit_hours", "event"])

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        risk_score = fold_model.predict(X_val)
        pred_surv = get_surv_pred_from_model(fold_model, X_val, horizons)

        # Independent OOF inside this repeat
        # In one repeat, each sample appears in validation exactly once.
        repeat_oof_risk[val_idx] = risk_score
        repeat_oof_surv[val_idx] = pred_surv
        repeat_oof_count[val_idx] += 1

        # Aggregated OOF across repeats
        final_oof_risk_sum[val_idx] += risk_score
        final_oof_surv_sum[val_idx] += pred_surv
        final_oof_count[val_idx] += 1

        if verbose:
            print(
                f"Repeat [{repeat_idx + 1}/{n_repeats}] "
                f"Fold [{fold_in_repeat + 1}/{n_splits}] done"
            )

        # End of one repeat
        if (fold_in_repeat + 1) == n_splits:
            current_repeat = repeat_idx + 1

            # Every sample should have exactly one OOF prediction in this repeat
            assert np.all(repeat_oof_count == 1), (
                f"Unexpected repeat OOF counts at repeat {current_repeat}: "
                f"{np.unique(repeat_oof_count)}"
            )

            repeat_oof_results.append(
                {
                    "repeat": current_repeat,
                    "oof_risk": repeat_oof_risk.copy(),
                    "oof_surv": repeat_oof_surv.copy(),
                    "oof_count": repeat_oof_count.copy(),
                }
            )

            if verbose:
                print(
                    f"Completed repeat {current_repeat}/{n_repeats} | "
                    f"Repeat OOF count unique: {np.unique(repeat_oof_count)}"
                )

    final_oof_risk = final_oof_risk_sum / np.maximum(final_oof_count, 1)
    final_oof_surv = final_oof_surv_sum / np.maximum(final_oof_count[:, None], 1)

    assert np.all(final_oof_count == n_repeats), (
        f"Unexpected final OOF counts: {np.unique(final_oof_count)}"
    )

    return {
        "final_oof_risk": final_oof_risk,
        "final_oof_surv": final_oof_surv,
        "final_oof_count": final_oof_count,
        "repeat_oof_results": repeat_oof_results,
    }

#TODO OOF Prediction기반으로 수정 예정
def KFold_val(
    model: BaseEstimator,
    data: pd.DataFrame,
    seed,
    n_splits=5,
    n_repeats=2,
    verbose=False
) -> KFoldResult:
    
    horizons = np.asarray(HORIZONS, dtype=float).copy()
    horizons[-1] = min(horizons[-1], data['time_to_hit_hours'].max() - 1e-6)

    result = make_oof_predictions(
        model=model,
        data=data,
        horizons=horizons,
        seed=seed,
        n_splits=n_splits,
        n_repeats=n_repeats,
        verbose=verbose,
    )
    oof_list = result["repeat_oof_results"]

    y_full = make_surv_y(
        event=data['event'],
        time=data['time_to_hit_hours']
    )
    
    c_indices = []
    mean_briers = []
    hybrid_scores = []
    
    for repeat_result in oof_list:
        repeat_oof_risk = repeat_result["oof_risk"]
        repeat_oof_surv = repeat_result["oof_surv"]
        # Final score based on full OOF predictions
        oof_result = compute_hybrid_score(
            y_full, y_full, repeat_oof_risk, repeat_oof_surv, horizons
        )
        c_indices.append(oof_result.c_index)
        mean_briers.append(oof_result.mean_brier)
        hybrid_scores.append(oof_result.hybrid_score)

    c_indices = np.array(c_indices)
    mean_briers = np.array(mean_briers)
    hybrid_scores = np.array(hybrid_scores)
    
    return KFoldResult(
        c_index=np.mean(c_indices).item(),
        mean_brier=np.mean(mean_briers).item(),
        hybrid_score=np.mean(hybrid_scores).item(),
        std_c_index=np.std(c_indices).item(),
        std_mean_brier=np.std(mean_briers).item(),
        std_hybrid=np.std(hybrid_scores).item(),
    )
    
def calc_pred_corr(pred1: np.ndarray, pred2: np.ndarray) -> float:
    return np.corrcoef(pred1.ravel(), pred2.ravel())[0, 1]
    
def find_ensemble_model(
    oof_result: dict[int, dict],
    label=None,
    max_pair_corr: float = 0.995,
    max_ensemble_corr: float = 0.995,
    min_imporvement_score: float = 0.0003,
    max_model_num: int = 10,
    init_model_list: list = None,
    horizons: list = None,
    eps: float = 1e-6,
    allow_duplicate: bool = False,
    max_select: int = None,
    max_select_ratio: float = 0.3,
    verbose: bool = False,
) -> EnsembleModel:

    if init_model_list is None:
        init_model_list = [106]

    if horizons is None:
        horizons = [12, 24, 48, 72]

    if max_select is None:
        max_select = max(1, int(max_model_num * max_select_ratio))

    full_model_list = list(oof_result.keys())
    select_model_list = init_model_list.copy()

    if allow_duplicate:
        candidate_model_list = full_model_list.copy()
    else:
        candidate_model_list = [m for m in full_model_list if m not in select_model_list]

    select_oof_surv_list = [oof_result[trial_id]["oof_surv"] for trial_id in select_model_list]
    select_oof_risk_list = [oof_result[trial_id]["oof_risk"] for trial_id in select_model_list]

    eval_horizons = horizons.copy()
    eval_horizons[-1] = min(72, label["time"].max() - eps)

    prev_eval_result = compute_hybrid_score(
        label,
        label,
        np.mean(np.stack(select_oof_risk_list, axis=0), axis=0),
        np.mean(np.stack(select_oof_surv_list, axis=0), axis=0),
        eval_horizons,
    )

    if verbose:
        print("===== Initial Ensemble =====")
        print(f"Initial models: {select_model_list}")
        print(f"Initial hybrid score: {prev_eval_result.hybrid_score:.6f}")
        print(f"Allow duplicate: {allow_duplicate}")
        print(f"Max select per model: {max_select}")

    while len(select_model_list) < max_model_num:

        if verbose:
            print("\n==============================")
            print(f"Current ensemble size: {len(select_model_list)}")
            print(f"Current hybrid score: {prev_eval_result.hybrid_score:.6f}")
            print("==============================")

        max_improvement = 0.0
        select_trial_id = None
        max_score_result = None

        current_ensemble_risk = np.mean(np.stack(select_oof_risk_list, axis=0), axis=0)
        current_ensemble_surv = np.mean(np.stack(select_oof_surv_list, axis=0), axis=0)

        for trial_num in candidate_model_list:
            one_trial_oof_result = oof_result[trial_num]
            candidate_risk = one_trial_oof_result["oof_risk"]
            candidate_surv = one_trial_oof_result["oof_surv"]

            is_duplicate = trial_num in select_model_list

            # Duplicate count limit
            if allow_duplicate and select_model_list.count(trial_num) >= max_select:
                if verbose:
                    print(f"[Trial {trial_num}] skip (duplicate count limit: {max_select})")
                continue

            # Candidate vs selected each model
            pair_corrs = [
                calc_pred_corr(candidate_risk, selected_risk)
                for selected_risk in select_oof_risk_list
            ]
            max_pair_corr_val = max(pair_corrs) if len(pair_corrs) > 0 else 0.0

            # Candidate vs current ensemble mean
            ensemble_corr_val = calc_pred_corr(candidate_risk, current_ensemble_risk)

            # Correlation filtering:
            # - If duplicate is not allowed: always filter
            # - If duplicate is allowed: only filter for new models, not repeated picks
            if max_pair_corr_val > max_pair_corr:
                if verbose:
                    print(
                        f"[Trial {trial_num}] skip "
                        f"(pair corr={max_pair_corr_val:.5f} > {max_pair_corr})"
                    )
                continue

            if ensemble_corr_val > max_ensemble_corr:
                if verbose:
                    print(
                        f"[Trial {trial_num}] skip "
                        f"(ensemble corr={ensemble_corr_val:.5f} > {max_ensemble_corr})"
                    )
                continue

            ensemble_pred_mat_risk = np.stack(
                select_oof_risk_list + [candidate_risk], axis=0
            )
            ensemble_pred_mat_surv = np.stack(
                select_oof_surv_list + [candidate_surv], axis=0
            )

            eval_result = compute_hybrid_score(
                label,
                label,
                np.mean(ensemble_pred_mat_risk, axis=0),
                np.mean(ensemble_pred_mat_surv, axis=0),
                eval_horizons,
            )

            improve_score = eval_result.hybrid_score - prev_eval_result.hybrid_score

            if verbose:
                print(
                    f"[Trial {trial_num}] "
                    f"score={eval_result.hybrid_score:.6f} "
                    f"improve={improve_score:+.6f} "
                    f"pair_corr={max_pair_corr_val:.5f} "
                    f"ens_corr={ensemble_corr_val:.5f} "
                    f"duplicate={is_duplicate}"
                )

            if improve_score < min_imporvement_score:
                continue

            if max_improvement < improve_score:
                select_trial_id = trial_num
                max_improvement = improve_score
                max_score_result = eval_result

        if select_trial_id is None:
            print("\n========================")
            print("No candidate model satisfies the conditions.")
            print(f"Max Pair Corr: {max_pair_corr}")
            print(f"Max Ensemble Corr: {max_ensemble_corr}")
            print(f"Min Improve Score: {min_imporvement_score}")
            print("Stopping the search.")
            print("========================\n")
            break

        print(f"\n>>> SELECTED MODEL [{len(select_model_list) + 1} / {max_model_num}]")
        print(f"Trial: {select_trial_id}")
        print(f"Improvement: {max_improvement:+.6f}")
        print(f"New hybrid score: {max_score_result.hybrid_score:.6f}")

        select_model_list.append(select_trial_id)

        if (not allow_duplicate) or (select_model_list.count(select_trial_id) >= max_select):
            if select_trial_id in candidate_model_list:
                candidate_model_list.remove(select_trial_id)

        select_oof_surv_list.append(oof_result[select_trial_id]["oof_surv"])
        select_oof_risk_list.append(oof_result[select_trial_id]["oof_risk"])

        prev_eval_result = max_score_result

    model_counter = Counter(select_model_list)
    total_count = sum(model_counter.values())

    model_weights = {
        model_id: count / total_count
        for model_id, count in model_counter.items()
    }

    if verbose:
        print("\n===== Final Ensemble =====")
        print(f"Selected models: {select_model_list}")
        print(f"Model counts: {dict(model_counter)}")
        print(f"Model weights: {model_weights}")
        print(f"Final hybrid score: {prev_eval_result.hybrid_score:.6f}")

    return EnsembleModel(
        model_weights=model_weights,
    )

def make_corr_matrix(pred_matrix:np.ndarray, flatten=True) -> np.ndarray:
    """
    Compute correlation matrix between models.

    Args:
        pred_matrix (np.ndarray): 
            Shape = (n_models, n_samples, n_horizons)

        flatten (bool) :
            True  -> flatten (samples × horizons)
            False -> horizon-wise correlation averaged

    Returns:
        corr_matrix : np.ndarray
            Shape = (n_models, n_models)
    """
    
    n_models, n_samples, n_horizons = pred_matrix.shape

    if flatten:
        flat = pred_matrix.reshape(n_models, -1)
        corr_matrix = np.corrcoef(flat, rowvar=True)
        return np.nan_to_num(corr_matrix)

    corr_by_horizon = np.zeros((n_horizons, n_models, n_models), dtype=float)

    for h in range(n_horizons):
        horizon_preds = pred_matrix[:, :, h]   # (n_models, n_samples)
        corr_h = np.corrcoef(horizon_preds, rowvar=True)
        corr_by_horizon[h] = np.nan_to_num(corr_h)

    return corr_by_horizon
    


def create_features(df: pd.DataFrame, eps=0.1, max_hours=9999, min_speed=0.01)-> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): Input DataFrame with raw features.
        eps (float): Small constant to avoid division by zero in transformations.
        max_hours (int): Maximum hours to cap time-based features for stability.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
        
    ---
   

    ### 
    """
    one_km = 1000
    one_ha = 10000
    result = df.copy()
    dist = result['dist_min_ci_0_5h'].clip(lower=1)
    speed = result['closing_speed_m_per_h']
    perimeters = result['num_perimeters_0_5h']
    area_first = result['area_first_ha']
    result['log_distance'] = np.log1p(dist)
    result['inv_distance'] = 1 / (dist / one_km + eps)
    result['inv_distance_sq'] = result['inv_distance'] ** 2
    result['sqrt_distance'] = np.sqrt(dist)
    result['dist_km'] = dist / one_km
    result['dist_km_sq'] = (dist / one_km) ** 2
    result['dist_rank'] = dist.rank(pct=True)
    fire_radius = np.sqrt(area_first * one_ha / np.pi)
    result['radius_to_dist'] = fire_radius / dist
    result['area_to_dist_ratio'] = area_first / (dist / one_km + eps)
    result['log_area_dist_ratio'] = np.log1p(area_first) - np.log1p(dist)
    result['has_movement'] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    result['eta_hours'] = np.where(closing_pos > min_speed, dist / closing_pos, max_hours).clip(max=max_hours)
    result['log_eta'] = np.log1p(result['eta_hours'].clip(0, max_hours))
    radial_growth = result['radial_growth_rate_m_per_h'].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    result['effective_closing_speed'] = effective_closing
    result['eta_effective'] = np.where(effective_closing > min_speed, dist / effective_closing, max_hours).clip(max=max_hours)
    result['threat_score'] = result['alignment_abs'] * speed / np.log1p(dist)
    result['fire_urgency'] = perimeters * speed
    result['growth_intensity'] = result['area_growth_rate_ha_per_h'] * perimeters
    result['zone_critical'] = (dist < 5000).astype(float)
    result['zone_warning'] = ((dist >= 5000) & (dist < 10000)).astype(float)
    result['zone_safe'] = (dist >= 10000).astype(float)
    result['is_summer'] = result['event_start_month'].isin([6, 7, 8]).astype(float)
    result['is_afternoon'] = ((result['event_start_hour'] >= 12) & (result['event_start_hour'] < 20)).astype(float)
    drop_cols = ['relative_growth_0_5h', 'projected_advance_m', 'centroid_displacement_m',
                 'centroid_speed_m_per_h', 'closing_speed_abs_m_per_h', 'area_growth_abs_0_5h']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result