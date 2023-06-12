import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def BAG(train_pred, test_pred, train_y, test_y):
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

    References:
    Bias-adjustment in neuroimaging-based brain age frameworks: A robust scheme
    """

    line_reg = LinearRegression()
    poly = PolynomialFeatures(degree=1, include_bias=True)
    line_reg_features = poly.fit_transform(train_y.reshape(-1, 1))
    offset_train = train_pred - train_y
    line_reg.fit(line_reg_features, offset_train.reshape(-1, 1))
    test_features = poly.transform(test_y.reshape(-1, 1))
    offset_test = np.array(line_reg.predict(test_features)).flatten()
    corrected_age = test_y + offset_test

    return corrected_age, offset_train
