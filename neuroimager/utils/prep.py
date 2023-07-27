from sklearn.linear_model import LinearRegression


def rm_covars_linear(features, covariates):
    """
    Remove multiple covariates from the feature set with linear regression.

    Parameters
    ----------
    features : 2D numpy array, shape (n_samples, n_features_X)
        Set of features.
    covariates : 2D numpy array, shape (n_samples, n_covariates)
        Covariates to regress out.

    Returns
    -------
    new_features : 2D numpy array, shape (n_samples, n_features_X)
        Residuals of the features after regressing out the covariates.
    """
    model = LinearRegression()
    model.fit(covariates, features)
    new_features = features - model.predict(covariates)

    return new_features
