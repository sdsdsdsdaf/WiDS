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
try:
    from Utils.Config import Config, MetricOuput, KFoldResult
except ImportError:
    from Config import Config, MetricOuput, KFoldResult
from dataclasses import asdict


HORIZONS= np.array([12.0, 24.0, 48.0, 72.0])  # 예측할 시간 간격 (시간 단위)

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
def make_oof_predictions(model: BaseEstimator, data:pd.DataFrame, seed, n_splits=5, n_repeats=2, verbose=False, horizons=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    KFold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n_samples = len(data)
    n_horizons = len(horizons)

    oof_risk_sum = np.zeros(n_samples, dtype=float)
    oof_surv_sum = np.zeros((n_samples, n_horizons), dtype=float)
    oof_count = np.zeros(n_samples, dtype=int)
    
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(KFold.split(data)), total=n_splits * n_repeats, desc="KFold OOF", leave=False):
        X_train = data.iloc[train_idx].drop(columns=['time_to_hit_hours', 'event'])
        y_train = make_surv_y(
            event=data.iloc[train_idx]['event'],
            time=data.iloc[train_idx]['time_to_hit_hours']
        )

        X_val = data.iloc[val_idx].drop(columns=['time_to_hit_hours', 'event'])

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        risk_score = fold_model.predict(X_val)
        pred_surv = get_surv_pred_from_model(fold_model, X_val, horizons)

        oof_risk_sum[val_idx] += risk_score
        oof_surv_sum[val_idx] += pred_surv
        oof_count[val_idx] += 1

        if verbose:
            print(f'Fold [{fold + 1}/{n_splits * n_repeats}] done')

    oof_risk = oof_risk_sum / np.maximum(oof_count, 1.0)
    oof_surv = oof_surv_sum / np.maximum(oof_count[:, None], 1.0)
    
    assert np.all(oof_count == n_repeats), f"Unexpected OOF counts: {np.unique(oof_count)}"

    return oof_risk, oof_surv, oof_count

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


    oof_risk, oof_surv, oof_count = make_oof_predictions(
        model, data, seed, n_splits, n_repeats, verbose, horizons
    )

    y_full = make_surv_y(
        event=data['event'],
        time=data['time_to_hit_hours']
    )

    # Final score based on full OOF predictions
    oof_result = compute_hybrid_score(
        y_full, y_full, oof_risk, oof_surv, horizons
    )

    return KFoldResult(
        c_index=oof_result.c_index,
        mean_brier=oof_result.mean_brier,
        hybrid_score=oof_result.hybrid_score,
    )



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