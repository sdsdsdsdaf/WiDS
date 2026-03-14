from typing import Any

from sksurv.linear_model import CoxnetSurvivalAnalysis
import numpy
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

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
        