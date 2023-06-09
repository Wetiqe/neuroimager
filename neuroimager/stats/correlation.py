import numpy as np
import pingouin as pg
import pandas as pd


def partial_corrs(
    data,
    x_labels: list = None,
    y_labels: list = None,
    covars: list = None,
    method="auto",
    semi=False,
    alternative="two-sided",
    correct_p="global",
    correct_method="holm",
):
    """
    Compute partial correlations between pairs of variables, controlling for specified covariates.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        A 2D Pandas DataFrame or NumPy Array containing the data.
    x_labels : list
        A list of column indices or names for the independent variables.
    y_labels : list
        A list of column indices or names for the dependent variables.
    covariates : list, optional
        A list of column indices or names for the covariates to control for. Default is None.
    method : str, optional
        The correlation method to use. Options are 'auto', 'pearson', or 'spearman'. Default is 'auto'.
    semi : bool or str, optional
        If True, compute semi-partial correlation.
        If 'x', control for covariates only in x.
        If 'y', control for covariates only in y. Default is False.
    alternative : str, optional
        The alternative hypothesis for the correlation test. Options are 'two-sided', 'greater', or 'less'.
        Default is 'two-sided'.
    correct_p : str, optional
        The correction method for multiple comparisons. Options are 'global', 'IVs', or any other value for no correction.
        Default is 'global'.
    correct_method : str, optional
        The method to use for multiple comparison correction.
        Options are 'holm', 'bonferroni', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky', 'sidak', or 'none'.
        Default is 'holm'.

    Returns
    -------
    r_matrix : pd.DataFrame
        A DataFrame containing the partial correlation coefficients between the independent and dependent variables.
    p_matrix : pd.DataFrame
        A DataFrame containing the p-values for the partial correlation coefficients.

    Notes
    -----
    This function computes partial correlations between pairs of variables, controlling for specified covariates.
    P-values are corrected for multiple comparisons using the specified method.
    The function supports both Pearson and Spearman correlation methods, as well as semi-partial correlations.

    References
    ----------
    1. Pingouin documentation: https://pingouin-stats.org/generated/pingouin.partial_corr.html
    2. Pingouin multiple comparison correction: https://pingouin-stats.org/generated/pingouin.multicomp.html
    """
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except:
            raise ValueError("data must be a 2D Pandas DataFrame or NumPy Array")
    if len(x_labels) == 0 or len(y_labels) == 0:
        raise ValueError("X and Y must be specified")
    if len(covars) == 0:
        print(
            "You did not specify any covariates, will proceed with simple corrlelation"
        )
    IVs = data.columns[x_labels]  # independent variables
    DVs = data.columns[y_labels]  # dependent variables
    r_matrix = pd.DataFrame(np.zeros((len(IVs), len(DVs))), index=IVs, columns=DVs)
    p_matrix = r_matrix.copy()

    for y_label in DVs:
        p_vals = []
        for x_label in IVs:
            # Create a df only contains the variables using now
            # This is a good practice when using pingouin
            df = data[[x_label, y_label] + covars]
            df = df.dropna(how="any", axis=0)  # drop any row with missing value

            # Select correlation method based on multivariate normality test result
            if method == "auto":
                method_to_use = (
                    "pearson"
                    if pg.multivariate_normality(df[[x_label, y_label]])[2]
                    else "spearman"
                )
            else:
                method_to_use = method
            if not semi:
                results = pg.partial_corr(
                    data=df,
                    x=x_label,
                    y=y_label,
                    covar=covars,
                    alternative=alternative,
                    method=method_to_use,
                ).round(3)
            elif semi == "x":
                results = pg.partial_corr(
                    data=df,
                    x=x_label,
                    y=y_label,
                    x_covar=covars,
                    alternative=alternative,
                    method=method_to_use,
                ).round(3)
            elif semi == "y":
                results = pg.partial_corr(
                    data=df,
                    x=x_label,
                    y=y_label,
                    y_covar=covars,
                    alternative=alternative,
                    method=method_to_use,
                ).round(3)

            r_matrix.loc[x_label, y_label] = results.loc[method_to_use, "r"]
            p_vals.append(results.loc[method_to_use, "p-val"])
        if correct_p == "IVs":
            bools, p_vals = pg.multicomp(p_vals, alpha=0.05, method=correct_method)
        p_matrix.loc[:, y_label] = p_vals
    if correct_p == "global":
        pvals = p_matrix.values.flatten()
        bools, p_vals = pg.multicomp(p_vals, alpha=0.05, method=correct_method)
        p_matrix = pd.DataFrame(
            p_vals.reshape(len(x_labels), len(y_labels)),
            index=p_matrix.index,
            columns=p_matrix.columns,
        )
    return r_matrix, p_matrix
