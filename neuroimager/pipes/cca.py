from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from neuroimager.plotting import density_scatter


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
        correlation_measure: Union[
            str, Callable
        ] = "pearson",  # TODO: implement callable
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
        self.corr = correlation_measure

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
            rs.append(pd.DataFrame(X_perm).corrwith(pd.DataFrame(Y_perm)))
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

    def transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the original loadings"""
        if hasattr(self, "pca_x_model"):
            X_loadings = self.pca_x_model.components_.T @ self.cca_model.x_loadings_
        else:
            X_loadings = self.cca_model.x_loadings_
        if hasattr(self, "pca_y_model"):
            Y_loadings = self.pca_y_model.components_.T @ self.cca_model.y_loadings_
        else:
            Y_loadings = self.cca_model.y_loadings_
        self.x_loadings_ = X_loadings
        self.y_loadings_ = Y_loadings
        return X_loadings, Y_loadings


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
    weights,
    labels,
    ax=False,
    display_thresh=0.2,
    show=False,
    min_font_size=6,
    scaling_factor=50,
    **kwargs,
):
    xlim = kwargs.get("xlim", (-1, 1))
    xlabel = kwargs.get("xlabel", "")
    title = kwargs.get("title", "")
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("white")
    weights = weights.flatten()
    # Sort weights and labels for X features
    sorted_indices = np.argsort(weights)
    sorted_weights = weights[sorted_indices]
    if labels is not None:
        sorted_labels = [labels[idx] for idx in sorted_indices]

    # Plot X feature weights
    ax.set_ylabel("Weights")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(-0.5, len(weights) + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # perform min-max scalling for weights
    abs_weights = abs(sorted_weights)
    plot_weights = (abs_weights - abs_weights.min()) / (
        abs_weights.max() - abs_weights.min()
    )
    texts = []
    for i, label in enumerate(sorted_labels):
        if abs(sorted_weights[i]) < display_thresh:
            continue
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
    weights_X,
    weights_Y,
    display_thresh=0.2,
    scaling_factor=30,
    x_labels=None,
    y_labels=None,
    **kwargs,
):
    """
    Plots Canonical Correlation Analysis (CCA) feature weights.

    Parameters:
    weights_X (array-like): Weights for X features.
    weights_Y (array-like): Weights for Y features.
    x_labels (list, optional): Labels for X features. Defaults to None.
    y_labels (list, optional): Labels for Y features. Defaults to None.

    Returns:
    None
    """
    figsize = kwargs.get("fig_size", (10, 5))
    n_comps = weights_X.shape[1]
    if x_labels:
        assert len(x_labels) == weights_X.shape[0]
    if y_labels:
        assert len(y_labels) == weights_Y.shape[0]

    for i in range(n_comps):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
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
        )
        plt.tight_layout()
        plt.show()
