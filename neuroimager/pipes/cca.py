import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from neuroimager.plotting import density_scatter


warnings.warn("The functions are improved and moved to a separate package: permcca")


class perm_CCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_comps: int = 1,
        n_perms: int = 1000,
        cca_kwargs: Optional[Dict] = None,
        pca_x: bool = False,
        pca_y: bool = False,
        pca_x_kwargs: Optional[Dict] = None,
        pca_y_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        n_comp : int, optional, default: 1
            Number of CCA components to use.
        n_perm : int, optional, default: 1000
            Number of permutations to perform.
        random_state : int, RandomState instance or None, optional, default: None
            Seed for random number generator.
        """
        self.n_comps = n_comps
        self.cca_kwargs = cca_kwargs or {}
        self.n_perms = n_perms
        self.random_state = random_state
        self.pca_x = pca_x
        self.pca_y = pca_y
        default_pca_kwargs = {"n_components": 0.95, "whiten": True}
        self.pca_x_kwargs = pca_x_kwargs if pca_x_kwargs else default_pca_kwargs
        self.pca_y_kwargs = pca_y_kwargs if pca_y_kwargs else default_pca_kwargs
        self.cca_model = None
        self.X_c_ = None
        self.Y_c_ = None
        self.orig_rs_ = None
        self.X_loadings_ = None
        self.Y_loadings_ = None

    def _pca_decompose(
        self, raw_x: np.ndarray, raw_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.pca_x:
            self.pca_x_model = PCA(**self.pca_x_kwargs)
            self.pca_x_model.fit(raw_x)
            x = self.pca_x_model.transform(raw_x)
        else:
            x = raw_x
        if self.pca_y:
            self.pca_y_model = PCA(**self.pca_y_kwargs)
            self.pca_y_model.fit(raw_y)
            y = self.pca_y_model.transform(raw_y)
        else:
            y = raw_y
        return x, y

    def _cca_decompose(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[CCA, np.ndarray, np.ndarray]:
        x, y = self._pca_decompose(x, y)
        cca = CCA(n_components=self.n_comps, *self.cca_kwargs)
        x_c, y_c = cca.fit_transform(x, y)
        return cca, x_c, y_c

    def _calc_p(self, perm_rs: pd.DataFrame, orig_rs: List[float]) -> List[float]:
        p_vals = []
        for i in range(self.n_comps):
            p_value = ((perm_rs.iloc[:, i]) > abs(orig_rs[i])).sum() / perm_rs.shape[0]
            p_vals.append(p_value)

        return p_vals

    def _permute(self, X: np.ndarray, Y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        rs = []
        for _ in range(self.n_perms):
            ids = rng.permutation(X.shape[0])
            x_perm = X[ids, :]
            cca_perm, X_perm, Y_perm = self._cca_decompose(x_perm, Y)
            rs.append(
                pd.DataFrame(X_perm).corrwith(
                    pd.DataFrame(Y_perm), method=self.corr_method
                )
            )
        perm_rs = pd.DataFrame(rs)

        return perm_rs

    def permutation_test_cca(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[CCA, np.ndarray, np.ndarray, List[float], List[float]]:
        """
        Perform a permutation test for Canonical Correlation Analysis (CCA) to assess the significance
        of the correlation between two sets of features.

        Parameters
        ----------
        X : 2D numpy array, shape (n_samples, n_features_X)
            First set of features.
        Y : 2D numpy array, shape (n_samples, n_features_Y)
            Second set of features.
        Returns
        -------
        cca : CCA object
            Fitted CCA model.
        X_c : Transformed X data from CCA.
        Y_c : Transformed Y data from CCA.
        original_correlations : list of float
            Correlation values between the two linear combinations of features.
        p_values : list of float
            List of p-values for each CCA component.
        """
        np.random.seed(self.random_state)

        # Fit the CCA model
        cca, X_c, Y_c = self._cca_decompose(X, Y)

        # Compute the original correlations
        orig_rs = [pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(self.n_comps)]

        # Perform the permutation test
        perm_rs = self._permute(X, Y)
        p_values = self._calc_p(perm_rs, orig_rs)
        self.cca_model = cca
        self.X_c_ = X_c
        self.Y_c_ = Y_c
        self.orig_rs_ = orig_rs
        return cca, X_c, Y_c, orig_rs, p_values

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fits the CCA model on input data."""
        np.random.seed(self.random_state)

        # Fit CCA
        cca, X_c, Y_c = self._cca_decompose(X, Y)
        orig_rs = [pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(self.n_comps)]

        # Store the fitted CCA and transformed data as class attributes
        self.cca_model = cca
        self.X_c_ = X_c
        self.Y_c_ = Y_c
        self.orig_rs_ = orig_rs
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray):
        """Return the original loadings"""
        if hasattr(self, "pca_x_modarraydel"):
            X_loadings = self.pca_x_model.components_.T @ self.cca_model.x_loadings_
            X_loadings = self.cca_model.x_loadings_
        else:
            X_loadings = self.cca_model.x_loadings_
        if hasattr(self, "pca_y_model"):
            Y_loadings = self.pca_y_model.components_.T @ self.cca_model.y_loadings_
        else:
            Y_loadings = self.cca_model.y_loadings_
        self.x_loadings_ = X_loadings
        self.y_loadings_ = Y_loadings
        return X_loadings, Y_loadings

    def explained_variance_ratio(self) -> Tuple[List[float], List[float]]:
        """
        Calculate the explained variance ratio for both X and Y canonical variates.

        Returns
        -------
        explained_variance_ratio_X: list of float
            Explained variance ratio for each component in the X canonical variates.
        explained_variance_ratio_Y: list of float
            Explained variance ratio for each component in the Y canonical variates.
        """
        if self.X_c_ is None or self.Y_c_ is None:
            raise RuntimeError(
                "Please fit the model before calculating explained variance ratio."
            )

        # Calculate total variance for X and Y canonical variates
        total_variance_X = np.sum(np.var(self.X_c_, axis=0))
        total_variance_Y = np.sum(np.var(self.Y_c_, axis=0))

        # Calculate explained variance ratio for each component
        explained_variance_ratio_X = [
            np.var(self.X_c_[:, i]) / total_variance_X for i in range(self.n_comps)
        ]
        explained_variance_ratio_Y = [
            np.var(self.Y_c_[:, i]) / total_variance_Y for i in range(self.n_comps)
        ]

        return explained_variance_ratio_X, explained_variance_ratio_Y


def plot_cca_scatter(transformed_X, transformed_Y, n_comps=1, **kwargs):
    """
    Plots Canonical Correlation Analysis (CCA) scatter plot.

    Parameters:
    transformed_X (array-like): Transformed X data from CCA.
    transformed_Y (array-like): Transformed Y data from CCA.
    n_comps (int): Number of components to plot.

    Returns:
    None
    """
    fig_size = kwargs.get("fig_size", (10, 10))
    marker = kwargs.get("marker", "o")
    color = kwargs.get("color", "b")
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title_prefix = kwargs.get("title_prefix", "Scatter plot of CCA components")
    layout = kwargs.get("layout", "tight")
    p_values = kwargs.get("p_values", None)
    for i in range(n_comps):
        # plt.scatter(transformed_X[:, i], transformed_Y[:, i], marker=marker, color=color, label=f'Component {i+1}')
        density_scatter(transformed_X[:, i], transformed_Y[:, i], display=False)
        corr_coef = np.corrcoef(transformed_X[:, i], transformed_Y[:, i])[0, 1]
        if kwargs.get("xlabel", False) & kwargs.get("ylabel", False):
            title = f"{title_prefix}: {xlabel} vs {ylabel} pearson r = {corr_coef}"
            if p_values:
                title += f" p = {p_values[i]}"
            plt.title(title)
        else:
            title = f"Component {i + 1}: pearson r = {corr_coef}"
            if p_values:
                title += f" p = {p_values[i]}"
            plt.title(title)
        if not xlabel:
            xlabel = "X Component {}".format(i + 1)
        if not ylabel:
            ylabel = "Y Component {}".format(i + 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()


def plot_cca_weights(
    weights: List or np.ndarray,
    labels: List or np.ndarray,
    ax=False,
    display_thresh=0.2,
    show=False,
    min_font_size=6,
    scaling_factor=50,
    **kwargs,
):
    if isinstance(weights, list):
        weights = np.array(weights)
    elif isinstance(weights, np.ndarray):
        pass
    else:
        raise ValueError("weights must be a list or numpy array")
    if isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, np.ndarray):
        pass
    else:
        raise ValueError("labels must be a list or numpy array")

    xlim = kwargs.get("xlim", (-1, 1))
    xlabel = kwargs.get("xlabel", "")
    title = kwargs.get("title", "")
    display_title = kwargs.get("display_title", True)
    display_label = kwargs.get("display_label", True)
    sns.set_style("white")
    weights = weights.flatten()
    # get the index of thresholded weights
    idx = np.where(abs(weights) >= display_thresh)[0]
    filtered_weights = weights[idx]
    filtered_labels = labels[idx]
    # Sort weights and labels for X features
    sorted_indices = np.argsort(filtered_weights)
    sorted_weights = filtered_weights[sorted_indices]
    sorted_labels = [filtered_labels[idx] for idx in sorted_indices]
    if not ax:
        fig, ax = plt.subplots(figsize=(8, len(sorted_labels) / 2))
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(-0.5, len(weights) + 0.5)
    if display_label:
        ax.set_ylabel("Weights")
        ax.set_xlabel(xlabel)
    if display_title:
        ax.set_title(title)
    # perform min-max scalling for weights
    abs_weights = abs(sorted_weights)
    plot_weights = (abs_weights - abs_weights.min()) / (
        abs_weights.max() - abs_weights.min()
    )
    texts = []
    for i, label in enumerate(sorted_labels):
        font_size = max(min_font_size, plot_weights[i] * scaling_factor)
        ax.text(
            0,
            i,
            label + f"  weights {round(sorted_weights[i], 2)}",
            ha="center",
            va="center",
            fontsize=font_size,
        )

    # adjust_text(texts,only_move={ 'text':'X'})
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()

    return ax


def plot_paired_cca_weights(
    weights_X: np.ndarray or pd.DataFrame,
    weights_Y: np.ndarray or pd.DataFrame,
    x_labels: List = None,
    y_labels: List = None,
    display_thresh: float = 0.2,
    scaling_factor: float = 30,
    subplot_kwargs: dict = None,
):
    """
    Plots Canonical Correlation Analysis (CCA) feature weights.

    Parameters:
    weights_X (array-like): Weights for X features. If passing dataframe, the index represents the feature names.
    weights_Y (array-like): Weights for Y features. If passing dataframe, the index represents the feature names.
    x_labels (list): Labels for X features. Defaults to None. None only works with DataFrame.
    y_labels (list): Labels for Y features. Defaults to None. None only works with DataFrame.
    display_thresh (float): Threshold for displaying feature weights. Defaults to 0.2.
    scaling_factor (float): Scaling factor for font size of displayed weights. Defaults to 30.
    subplot_kwargs (dict, Optional): Default is None, passing to plot_cca_weights() function.
    **kwargs: Additional arguments Controls the figures.
    Returns:
    None
    """
    if isinstance(weights_X, pd.DataFrame):
        x_labels = weights_X.index
        weights_X = weights_X.values
    elif isinstance(weights_X, np.ndarray):
        pass
    else:
        raise TypeError("weights_X must be a numpy array or pandas dataframe.")
    if isinstance(weights_Y, pd.DataFrame):
        y_labels = weights_Y.index
        weights_Y = weights_Y.values
    elif isinstance(weights_Y, np.ndarray):
        pass
    else:
        raise TypeError("weights_Y must be a numpy array or pandas dataframe.")
    if x_labels is not None:
        if not isinstance(x_labels, list):
            try:
                x_labels = list(x_labels)
            except:
                raise TypeError("x_labels must be a list or can be converted to list.")
        if len(x_labels) != weights_X.shape[0]:
            raise ValueError(
                "The length of x_labels must be equal to the number of rows in weights_X"
            )
    else:
        raise ValueError("x_labels must be provided. What are you plotting then?")
    if y_labels is not None:
        if not isinstance(y_labels, list):
            try:
                y_labels = list(y_labels)
            except:
                raise TypeError("y_labels must be a list or can be converted to list.")
        if len(y_labels) != weights_Y.shape[0]:
            raise ValueError(
                "The length of y_labels must be equal to the number of rows in weights_Y"
            )
    else:
        raise ValueError("x_labels must be provided. What are you plotting then?")
    n_comps = weights_X.shape[1]
    for i in range(n_comps):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_cca_weights(
            weights_X[:, i],
            x_labels,
            ax=ax1,
            display_thresh=display_thresh,
            min_font_size=6,
            scaling_factor=scaling_factor,
            xlim=(-1, 1),
            xlabel="X Features",
            title="Component {} X feature weights".format(i + 1),
            **subplot_kwargs,
        )
        plot_cca_weights(
            weights_Y[:, i],
            y_labels,
            ax=ax2,
            display_thresh=display_thresh,
            min_font_size=6,
            scaling_factor=scaling_factor,
            xlabel="Y Features",
            title="Component {} Y feature weights".format(i + 1),
            **subplot_kwargs,
        )
        plt.tight_layout()
        plt.show()
