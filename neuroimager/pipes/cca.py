import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from neuroimager.plotting import density_scatter


def permutation_test_cca(X, Y, n_components=1, n_permutations=1000, random_state=None):
    """
    Perform a permutation test for Canonical Correlation Analysis (CCA) to assess the significance
    of the correlation between two sets of features.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features_X)
        First set of features.
    Y : 2D numpy array, shape (n_samples, n_features_Y)
        Second set of features.
    n_components : int, optional, default: 1
        Number of CCA components to use.
    n_permutations : int, optional, default: 1000
        Number of permutations to perform.
    random_state : int, RandomState instance or None, optional, default: None
        Seed for random number generator.

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
    np.random.seed(random_state)

    # Fit the CCA model
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)

    # Compute the original correlations
    original_correlations = [
        pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(n_components)
    ]

    # Perform the permutation test
    count_better_correlations = [0] * n_components
    seeds = np.random.randint(0, 100000, n_permutations)
    rs = []
    p_values = []
    for i, seed in enumerate(seeds):
        np.random.seed(seed)

        ids = np.random.permutation(X.shape[0])
        x_perm = X[ids, :]
        cca_perm = CCA(n_components=n_components)
        X_perm, Y_perm = cca_perm.fit_transform(x_perm, Y)
        rs.append(pd.DataFrame(X_perm).corrwith(pd.DataFrame(Y_perm)))
    r_permuted_values = pd.DataFrame(rs)
    for i in range(n_components):
        p_value = (
            abs(r_permuted_values.iloc[:, i]) > abs(original_correlations[i])
        ).sum() / r_permuted_values.shape[0]
        p_values.append(p_value)

    return cca, X_c, Y_c, original_correlations, p_values


def plot_cca_scatter(transformed_X, transformed_Y, n_components=1, **kwargs):
    """
    Plots Canonical Correlation Analysis (CCA) scatter plot.

    Parameters:
    transformed_X (array-like): Transformed X data from CCA.
    transformed_Y (array-like): Transformed Y data from CCA.
    n_components (int): Number of components to plot.

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
    for i in range(n_components):
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

    texts = []
    for i, label in enumerate(sorted_labels):
        if abs(sorted_weights[i]) < display_thresh:
            continue
        font_size = max(min_font_size, abs(sorted_weights[i]) * scaling_factor)
        ax.text(
            0,
            i,
            label + f"  weights {round(sorted_weights[i],2)}",
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
    n_components = weights_X.shape[1]

    for i in range(n_components):
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
