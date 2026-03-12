from dataclasses import dataclass, field
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray


@dataclass
class MetricOuput:
    c_index: float = 0.0
    mean_brier: float = 0.0
    hybrid_score: float = 0.0
    
@dataclass
class KFoldResult:
    c_index:float = 0.0
    mean_brier:float = 0.0
    hybrid_score:float = 0.0
    std_c_index:float = 0.0
    std_mean_brier:float = 0.0
    std_hybrid:float = 0.0

@dataclass
class EnsembleModel:
    model_weights:dict[int, float]
    model_type:str = "GBSA"


@dataclass
class GBSAConfig:
    # core boosting
    loss:str = "coxph"
    n_estimators: int = 100
    learning_rate: float = 0.1
    subsample: float = 1.0

    # tree structure
    max_depth: int = 3
    max_features: int = None
    max_leaf_nodes: int = None

    # split control
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    min_impurity_decrease: float = 0.0

    # tree criterion
    criterion: str = "friedman_mse"

    # regularization
    ccp_alpha: float = 0.0
    dropout_rate: float = 0.0

    # early stopping
    validation_fraction: float = 0.1
    n_iter_no_change: int = None
    tol: float = 1e-4

    # misc
    random_state: int = 42
    warm_start: bool = False
    verbose: int = 0

@dataclass
class PreprocessingConfig:
    eps: float = 1e-6
    min_speed: float = 0.01
    max_hours: float = 9999.0

@dataclass
class Config:
    seed: int = 42
    cv_n_splits: int = 5
    cv_n_repeats: int = 4
    gbsa_config: GBSAConfig = field(default_factory=lambda: GBSAConfig())
    preprocessing_config: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig())

@dataclass
class TrialResult:
    trial_id: int
    config: Config
    result: KFoldResult