from collections import Counter
import os
from numpy.typing import NDArray
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import numpy as np
from sksurv.util import Surv
from tqdm.auto import tqdm

try:
    from Utils.Config import Config, EnsembleModel, MetricOuput
    from Utils.utils import (
        load_config_yaml,
        set_seed,
        get_top_trial_oofs,
        calc_pred_corr,
        compute_hybrid_score
    )

except ImportError:
    from Config import Config, EnsembleModel, MetricOuput
    from utils import (
        load_config_yaml,
        set_seed,
        get_top_trial_oofs,
        calc_pred_corr,
        compute_hybrid_score
    )

def load_experiment_config(
    seed: int,
    model_type: str,
    trials_root: str = "Trials",
) -> Config:
    model_type = model_type.lower()
    model_dir = os.path.join(trials_root, str(seed), model_type)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    trial_dirs = sorted(
        d for d in os.listdir(model_dir)
        if d.startswith("trial_") and os.path.isdir(os.path.join(model_dir, d))
    )

    if len(trial_dirs) == 0:
        raise FileNotFoundError(f"No trial directories found in: {model_dir}")

    config_path = os.path.join(model_dir, trial_dirs[0], "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return load_config_yaml(config_path)

def load_study_from_dir(
    seed: int,
    model_type: str,
    trials_root: str = "Trials",
) -> optuna.Study:
    model_type = model_type.lower()
    journal_path = os.path.join(
        trials_root,
        str(seed),
        f"{model_type}_journal.log",
    )

    storage = JournalStorage(
        JournalFileBackend(journal_path)
    )

    study = optuna.load_study(
        study_name=f"{model_type}_survival_seed_{seed}",
        storage=storage,
    )
    return study

def collect_one_model_top_oofs(
    seed: int,
    model_type: str,
    data: pd.DataFrame,
    horizons: np.ndarray,
    trials_root: str = "Trials",
    out_dir: str = "TOP_OOF",
    top_ratio: float = 0.3,
) -> dict[tuple[int, str, int], dict]:
    
    model_type = model_type.lower()
    set_seed(seed)

    config = load_experiment_config(
        seed=seed,
        model_type=model_type,
        trials_root=trials_root,
    )

    study = load_study_from_dir(
        seed=seed,
        model_type=model_type,
        trials_root=trials_root,
    )

    return get_top_trial_oofs(
        study=study,
        data=data,
        horizons=horizons,
        out_dir=out_dir,
        top_ratio=top_ratio,
        seed=config.seed,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        model_type=config.model_type.lower(),
    )
    
def evaluate_candidate_ensemble(
    prev_pred_risk:NDArray,
    prev_pred_surv:NDArray,
    candidate_risk:NDArray,
    candidate_surv:NDArray,
    prev_eval_result:MetricOuput,
    label:Surv,
    eval_horizons:list[float] | NDArray,
    n_selected_models:int,
    use_grid_search: bool = False,
    weight_grid:list[float] | NDArray=None,
):
    """
    Evaluate one candidate model against current ensemble.

    Args:
        prev_pred_risk (np.ndarray) : 
            Current ensemble risk prediction, shape (n_samples,)
        prev_pred_surv (np.ndarray): 
            Current ensemble survival prediction, shape (n_samples, n_horizons)
        candidate_risk (np.ndarray): 
            Candidate model risk prediction
        candidate_surv (np.ndarray): 
            Candidate model survival prediction
        prev_eval_result (Utils.Config.MetricOuput): 
            Previous ensemble evaluation result. Must have `.hybrid_score`
        label (sksurv.Surv) : structured array or compatible object
            Survival label
        eval_horizons (list or np.ndarray): list or np.ndarray
            Evaluation horizons
        use_grid_search : bool
            If True, search best alpha from weight_grid.
            If False, use simple 0.5 / 0.5 mean with current ensemble.
        weight_grid (iterable, optional): 
            Candidate alpha values for:
                new_pred = alpha * prev_pred + (1 - alpha) * candidate_pred

    Returns:
        eval_result (Utils.Config.MetricOutput): 
            Result from compute_hybrid_score
        new_pred_risk (np.ndarray): 
            New ensembled risk prediction
        new_pred_surv (np.ndarray): 
            New ensembled survival prediction
        improve_score (float): 
            eval_result.hybrid_score - prev_eval_result.hybrid_score
        best_weight (float | None): 
            Best alpha when use_grid_search=True, else None
    """
    
    if not use_grid_search:
        new_pred_risk = (prev_pred_risk*n_selected_models + candidate_risk) / (n_selected_models+1)
        new_pred_surv = (prev_pred_surv*n_selected_models + candidate_surv) / (n_selected_models+1)

        eval_result = compute_hybrid_score(
            label,
            label,
            new_pred_risk,
            new_pred_surv,
            eval_horizons,
        )
        improve_score = eval_result.hybrid_score - prev_eval_result.hybrid_score

        return eval_result, new_pred_risk, new_pred_surv, improve_score, None

    if weight_grid is None:
        raise ValueError("weight_grid must be provided when use_grid_search=True")

    best_eval_result = None
    best_pred_risk = None
    best_pred_surv = None
    best_improve_score = -np.inf
    best_weight = None

    for alpha in weight_grid:
        new_pred_risk = alpha * prev_pred_risk + (1.0 - alpha) * candidate_risk
        new_pred_surv = alpha * prev_pred_surv + (1.0 - alpha) * candidate_surv

        eval_result = compute_hybrid_score(
            label,
            label,
            new_pred_risk,
            new_pred_surv,
            eval_horizons,
        )
        improve_score = eval_result.hybrid_score - prev_eval_result.hybrid_score

        if improve_score > best_improve_score:
            best_eval_result = eval_result
            best_pred_risk = new_pred_risk
            best_pred_surv = new_pred_surv
            best_improve_score = improve_score
            best_weight = float(alpha)

    return (
        best_eval_result,
        best_pred_risk,
        best_pred_surv,
        best_improve_score,
        best_weight,
    )

def collect_top_trial_oofs_from_configs(
    seeds: list[int],
    model_types: list[str],
    train_data: pd.DataFrame,
    horizons: np.ndarray,
    trials_root: str = "Trials",
    out_dir: str = "TOP_OOF",
    top_ratios: dict[str, float] | None = None,
    verbose: bool = True,
) -> dict[tuple[int, str, int], dict]:
    if top_ratios is None:
        top_ratios = {}

    all_oof_result: dict[tuple[int, str, int], dict] = {}

    for seed in seeds:
        for model_type in model_types:
            model_type = model_type.lower()
            top_ratio = top_ratios.get(model_type, 0.3)

            one_result = collect_one_model_top_oofs(
                seed=seed,
                model_type=model_type,
                data=train_data,
                horizons=horizons,
                trials_root=trials_root,
                out_dir=out_dir,
                top_ratio=top_ratio,
            )

            all_oof_result.update(one_result)

            if verbose:
                print(
                    f"[collect] seed={seed}, model={model_type}, "
                    f"added={len(one_result)}, total={len(all_oof_result)}"
                )

    return all_oof_result


def find_ensemble_model(
    oof_result: dict[tuple[str,int],dict],
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
    use_weight_grid_search:bool = False,
    weight_grid:NDArray = None,
) -> EnsembleModel:

    if use_weight_grid_search:
        allow_duplicate = False        
    
    if init_model_list is None:
        init_model_list = [(42,"gbsa", 160)]

    if horizons is None:
        horizons = [12, 24, 48, 72]

    if max_select is None:
        max_select = max(1, int(max_model_num * max_select_ratio))
    
    if not use_weight_grid_search:
        weight_grid = None
    
    if weight_grid is None and use_weight_grid_search:
        weight_grid = np.linspace(0.05, 0.95, 19)
        

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

    prev_ensemble_risk = np.mean(np.stack(select_oof_risk_list, axis=0), axis=0)
    prev_ensemble_surv = np.mean(np.stack(select_oof_surv_list, axis=0), axis=0)
    
    prev_eval_result = compute_hybrid_score(
        label,
        label,
        prev_ensemble_risk,
        prev_ensemble_surv,
        eval_horizons,
    )

    if verbose:
        print("===== Initial Ensemble =====")
        print(f"Initial models: {select_model_list}")
        print(f"Initial hybrid score: {prev_eval_result.hybrid_score:.6f}")
        print(f"Allow duplicate: {allow_duplicate}")
        print(f"Max select per model: {max_select}")
        print(f"Ensemble Candidate Model: {len(candidate_model_list)}")

    while len(select_model_list) < max_model_num:

        if verbose:
            print("\n==============================")
            print(f"Current ensemble size: {len(select_model_list)}")
            print(f"Current hybrid score: {prev_eval_result.hybrid_score:.6f}")
            print("==============================")

        max_improvement = 0.0
        select_trial_id = None
        max_score_result = None
        best_new_pred_risk = None
        best_new_pred_surv = None

        for trial_id in candidate_model_list:
            one_trial_oof_result = oof_result[trial_id]
            candidate_risk = one_trial_oof_result["oof_risk"]
            candidate_surv = one_trial_oof_result["oof_surv"]

            is_duplicate = trial_id in select_model_list

            # Duplicate count limit
            if allow_duplicate and select_model_list.count(trial_id) >= max_select:
                if verbose:
                    print(f"[Trial {trial_id}] skip (duplicate count limit: {max_select})")
                continue

            # Candidate vs selected each model
            pair_corrs = [
                calc_pred_corr(candidate_risk, selected_risk)
                for selected_risk in select_oof_risk_list
            ]
            max_pair_corr_val = max(pair_corrs) if len(pair_corrs) > 0 else 0.0

            # Candidate vs current ensemble mean
            ensemble_corr_val = calc_pred_corr(candidate_risk, prev_ensemble_risk)

            # Correlation filtering:
            # - If duplicate is not allowed: always filter
            # - If duplicate is allowed: only filter for new models, not repeated picks
            if max_pair_corr_val > max_pair_corr:
                if verbose:
                    print(
                        f"[Trial {trial_id}] skip "
                        f"(pair corr={max_pair_corr_val:.5f} > {max_pair_corr})"
                    )
                continue

            if ensemble_corr_val > max_ensemble_corr:
                if verbose:
                    print(
                        f"[Trial {trial_id}] skip "
                        f"(ensemble corr={ensemble_corr_val:.5f} > {max_ensemble_corr})"
                    )
                continue

            eval_result, new_pred_risk, new_pred_surv, improve_score, best_weight = evaluate_candidate_ensemble(
                prev_pred_risk=prev_ensemble_risk,
                prev_pred_surv=prev_ensemble_surv,
                candidate_risk=candidate_risk,
                candidate_surv=candidate_surv,
                prev_eval_result=prev_eval_result,
                label=label,
                eval_horizons=eval_horizons,
                use_grid_search=use_weight_grid_search,
                n_selected_models=len(select_model_list),
                weight_grid=weight_grid,
            )

            if verbose:
                msg = (
                    f"\n[Trial {trial_id}] "
                    f"\nscore={eval_result.hybrid_score:.6f} "
                    f"\nimprove={improve_score:+.6f} "
                    f"\npair_corr={max_pair_corr_val:.5f} "
                    f"\nens_corr={ensemble_corr_val:.5f} "
                    f"\nduplicate={is_duplicate}"
                )
                if use_weight_grid_search:
                    msg += f"\nbest_prev_weight={best_weight:.3f} \n best_candidate_weight={1.0-best_weight:.3f}"
                print(msg)

            if improve_score < min_imporvement_score:
                continue

            if max_improvement < improve_score:
                select_trial_id = trial_id
                max_improvement = improve_score
                max_score_result = eval_result
                best_new_pred_risk = new_pred_risk
                best_new_pred_surv = new_pred_surv

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
        prev_ensemble_risk = best_new_pred_risk
        prev_ensemble_surv = best_new_pred_surv

        prev_eval_result = max_score_result

    model_counter = Counter(select_model_list)
    total_count = sum(model_counter.values())

    model_weights = {
        model_id: count / total_count
        for model_id, count in model_counter.items()
    }

    print("\n===== Final Ensemble =====")
    print(f"Selected models: {select_model_list}")
    print(f"Model counts: {dict(model_counter)}")
    print(f"Model weights: {model_weights}")
    print(f"Final hybrid score: {prev_eval_result.hybrid_score:.6f}")
    print(f"Final C-index: {prev_eval_result.c_index:.6f}")
    print(f"Final mean brier: {prev_eval_result.mean_brier}")

    return EnsembleModel(
        model_weights=model_weights,
        ensemble_score=max_score_result
    )
    
def search_ensemble_weight(
    oof_result: dict[tuple[str,int],dict],
    model_dict: EnsembleModel,
    label,
    eval_horizons,
    weight_grid: NDArray | None = None,
    n_iter: int = 20000,
    random_state: int = 42,
) -> EnsembleModel:
    """
    Search optimal ensemble weights for selected models.

    Parameters
    ----------
    model_dict : EnsembleModel
        Contains model_weights keys (model ids) and OOF predictions.
    label : survival label
    eval_horizons : list
    weight_grid : optional grid (only used when model count == 2)
    n_iter : number of random samples for general case
    random_state : RNG seed
    """
    
    if model_dict is None or len(model_dict.model_weights) == 0:
        raise ValueError("No models found in model_dict.model_weights")
    
    rng = np.random.default_rng(random_state)

    model_list = list(model_dict.model_weights.keys())
    K = len(model_list)
    
    eval_horizons = eval_horizons.copy()
    eval_horizons[-1] = min(72, label["time"].max() - 1e-6)
    
    if K == 2 and weight_grid is None:
        weight_grid = np.linspace(0, 1, 1001)

    risk_preds = [oof_result[m]["oof_risk"] for m in model_list]
    surv_preds = [oof_result[m]["oof_surv"] for m in model_list]

    risk_stack = np.stack(risk_preds, axis=0)
    surv_stack = np.stack(surv_preds, axis=0)

    best_score = -np.inf
    best_weights = None
    best_result = None
    
    print(f'\nWeight Searching in {model_list}...')
    
        # ----- special case: 2 less models (exact grid) -----
    if K == 2 and weight_grid is not None:

        for w in weight_grid:
            weights = np.array([w, 1 - w])

            risk = np.tensordot(weights, risk_stack, axes=1)
            surv = np.tensordot(weights, surv_stack, axes=1)

            result = compute_hybrid_score(
                label,
                label,
                risk,
                surv,
                eval_horizons,
            )

            if result.hybrid_score > best_score:
                best_score = result.hybrid_score
                best_weights = weights
                best_result = result
                
    else:

        for _ in tqdm(range(n_iter)):

            weights = rng.dirichlet(np.ones(K))

            risk = np.tensordot(weights, risk_stack, axes=1)
            surv = np.tensordot(weights, surv_stack, axes=1)

            result = compute_hybrid_score(
                label,
                label,
                risk,
                surv,
                eval_horizons,
            )

            if result.hybrid_score > best_score:
                best_score = result.hybrid_score
                best_weights = weights
                best_result = result

    if best_weights is None:
        raise ValueError("Failed to find valid ensemble weights")
    weight_dict = {m: w for m, w in zip(model_list, best_weights)}

    return EnsembleModel(
        model_weights=weight_dict,
        ensemble_score=best_result
    )
    
    
    