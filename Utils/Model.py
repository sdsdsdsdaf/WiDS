from sksurv.linear_model import CoxnetSurvivalAnalysis
import numpy
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class CoxnetWithStandardScaler(BaseEstimator):
    def __init__(
        self,
        use_scaler=True,
        fit_baseline_model=True,
        l1_ratio=0.5,
        alpha_min_ratio=0.01,
        n_alphas=100,
        max_iter=100000,
        tol=1e-7,
        normalize=False,
        copy_X=True,
        verbose=False,
    ):
        self.use_scaler = use_scaler
        self.fit_baseline_model = fit_baseline_model
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

        self.scaler_ = None
        self.model_ = None

    def fit(self, X, y):
        X_cpy = X.copy()

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            X_cpy = self.scaler_.fit_transform(X_cpy)

        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alphas=self.n_alphas,
            max_iter=self.max_iter,
            tol=self.tol,
            normalize=self.normalize,
            copy_X=self.copy_X,
            verbose=self.verbose,
            fit_baseline_model=self.fit_baseline_model,
        )
        self.model_.fit(X_cpy, y)
        return self

    def predict(self, X):
        X_cpy = X.copy()
        if self.scaler_ is not None:
            X_cpy = self.scaler_.transform(X_cpy)
        return self.model_.predict(X_cpy)

    def predict_survival_function(self, X):
        X_cpy = X.copy()
        if self.scaler_ is not None:
            X_cpy = self.scaler_.transform(X_cpy)
        return self.model_.predict_survival_function(X_cpy)
        