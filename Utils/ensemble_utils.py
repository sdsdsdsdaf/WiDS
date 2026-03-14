import os
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import numpy as np

try:
    from Utils.Config import Config
    from Utils.utils import (
        load_config_yaml,
        set_seed,
        get_top_trial_oofs,
    )

except ImportError:
    from Config import Config
    from utils import (
        load_config_yaml,
        set_seed,
        get_top_trial_oofs,
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