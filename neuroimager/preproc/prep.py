import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class NuisanceLinear(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nuisance_variables: np.ndarray,
        standardize=True,
        line_reg_order=1,
        include_bias=True,
        line_reg_kwargs=None,
    ):
        self.line_reg_kwargs = {} if line_reg_kwargs is None else line_reg_kwargs
        self.line_reg = LinearRegression(**self.line_reg_kwargs)
        self.standardize = standardize
        self.include_bias = include_bias
        self.line_reg_order = line_reg_order

        self.nuisance_variables = nuisance_variables
        if len(self.nuisance_variables.shape) == 1:
            self.nuisance_variables = self.nuisance_variables.reshape(-1, 1)
        if self.standardize:
            scaler = StandardScaler()
            self.nuisance_variables = scaler.fit_transform(self.nuisance_variables)

        self.poly_nuisance_vars = PolynomialFeatures(
            degree=self.line_reg_order, include_bias=self.include_bias
        ).fit_transform(self.nuisance_variables)

    def fit(self, X, y=None):
        self.line_reg.fit(self.poly_nuisance_vars, X)
        return self

    def transform(self, X, y=None):
        predicted_X = self.line_reg.predict(self.poly_nuisance_vars)
        adjusted_X = X - predicted_X
        return adjusted_X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
