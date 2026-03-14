from dataclasses import dataclass, field
from typing import Any, TypedDict
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
    model_weights:dict[(int, str, int), float]
    ensemble_score:MetricOuput

@dataclass
class PreprocessingConfig:
    eps: float = 1e-6
    min_speed: float = 0.01
    max_hours: float = 9999.0

@dataclass
class Config:
    model_type: str = "gbsa"
    model_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    cv_n_splits: int = 5
    cv_n_repeats: int = 4
    preprocessing_config: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig())

@dataclass
class TrialResult:
    trial_id: int
    config: Config
    result: KFoldResult