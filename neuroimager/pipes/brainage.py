import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class BrainAgeGapCorrectionLinear(BaseEstimator, TransformerMixin):
    #     References:
    #     Bias-adjustment in neuroimaging-based brain age frameworks: A robust scheme
    def __init__(self, line_reg_order=1):
        self.line_reg_order = line_reg_order
        self.line_reg = LinearRegression()
        self.poly = PolynomialFeatures(degree=self.line_reg_order, include_bias=True)

    def fit(self, X, y):
        line_reg_features = self.poly.fit_transform(y.reshape(-1, 1))
        offset_train = X - y
        self.line_reg.fit(line_reg_features, offset_train.reshape(-1, 1))
        return self

    def transform(self, X, y):
        test_features = self.poly.transform(y.reshape(-1, 1))
        offset_test = np.array(self.line_reg.predict(test_features)).flatten()
        corrected_age = X - offset_test
        return corrected_age.reshape(-1, 1)


def BAG(train_pred, test_pred, train_y, test_y, line_reg_order=1):
    """
    Brain Age Gap Correction

    Parameters:
    train_pred (1D numpy array): Predicted age on the training set.
    test_pred (1D numpy array): Predicted age on the test set.
    train_y (1D numpy array): Actual age in the training set.
    test_y (1D numpy array): Actual age in the test set.

    Returns:
    corrected_age (numpy array): Corrected predicted age on the test set.
    train_delta_age (numpy array): Difference between predicted and actual age on the training set.
    line_reg (LinearRegression object): Trained regression model.

    """

    line_reg = LinearRegression()
    poly = PolynomialFeatures(degree=line_reg_order, include_bias=True)
    line_reg_features = poly.fit_transform(train_y.reshape(-1, 1))
    offset_train = train_pred - train_y
    line_reg.fit(line_reg_features, offset_train.reshape(-1, 1))
    test_features = poly.transform(test_y.reshape(-1, 1))
    offset_test = np.array(line_reg.predict(test_features)).flatten()
    corrected_age = test_pred - offset_test
    corrected_delta_age = corrected_age - test_y
    return corrected_age, corrected_delta_age
