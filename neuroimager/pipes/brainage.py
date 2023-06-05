import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def BAG(train_pred, test_pred, train_y, test_y):
    """
    Brain Age Gap Correction

    Parameters:
    train_pred (numpy array): Predicted age on the training set.
    test_pred (numpy array): Predicted age on the test set.
    train_y (numpy array): Actual age in the training set.
    test_y (numpy array): Actual age in the test set.

    Returns:
    corrected_age (numpy array): Corrected predicted age on the test set.
    train_delta_age (numpy array): Difference between predicted and actual age on the training set.
    line_reg (LinearRegression object): Trained regression model.
    """

    line_reg = LinearRegression()
    poly = PolynomialFeatures(degree=1, include_bias=True)
    line_reg_features = poly.fit_transform(train_y.reshape(-1, 1))
    line_reg.fit(line_reg_features, train_pred.reshape(-1, 1))

    test_features = poly.transform(test_y.reshape(-1, 1))
    corrected_delta_age = (
        test_pred - np.array(line_reg.predict(test_features)).flatten()
    )
    corrected_age = test_y + corrected_delta_age

    return corrected_age, corrected_delta_age
