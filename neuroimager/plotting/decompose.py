import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from typing import List, Union


# TODO: ADD Docstrings
def biplot_decompose(
    dim1: Union[List[float], np.ndarray],
    dim2: Union[List[float], np.ndarray],
    dim1_weights: Union[List[float], np.ndarray] = None,
    dim2_weights: Union[List[float], np.ndarray] = None,
    labels: List[str] = None,
    hue: List[str] = None,
    palette="plasma",
    ax=None,
    padding: float = 1.2,
    plot_weights=True,
    display: bool = True,
):
    dim1 = np.squeeze(np.array(dim1))
    dim2 = np.squeeze(np.array(dim2))
    dim1_weights = np.squeeze(np.array(dim1_weights))
    dim2_weights = np.squeeze(np.array(dim2_weights))
    to_check = [dim1, dim2]
    if plot_weights:
        to_check.extend([dim1_weights, dim2_weights])
    for input in to_check:
        if len(input.shape) > 1:
            raise ValueError("Input arrays must be 1-dimensional.")
        elif len(input.shape) < 1:
            raise ValueError("input can't be empty")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None
    if hue is None:
        sns.scatterplot(x=dim1, y=dim2, ax=ax, color="b")
    else:
        sns.scatterplot(x=dim1, y=dim2, ax=ax, palette=palette)
    ax.set_xlabel("Dimension 1", color="b")
    ax.set_ylabel("Dimension 2", color="b")
    ax.tick_params("y", colors="b")
    ax.tick_params("x", colors="b")

    xlim = max(abs(dim1)) * padding
    ylim = max(abs(dim2)) * padding
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    if not plot_weights:
        sns.despine(left=True, right=True, top=True, bottom=True)
        ax.tick_params(axis="both", which="both", length=0)
        if display:
            plt.show()
        return fig, ax

    ax2 = ax.twinx()
    loading_padding = padding + 0.1
    loading_ylim = np.max(np.abs(dim2_weights)) * loading_padding
    ax2.set_ylim(-loading_ylim, loading_ylim)
    ax2.set_ylabel("Dimension 2 weights", color="r")
    ax2.tick_params("y", colors="r")

    ax3 = ax2.twiny()
    loading_xlim = np.max(np.abs(dim1_weights)) * loading_padding
    ax3.set_xlim(-loading_xlim, loading_xlim)
    ax3.tick_params("x", colors="r")
    ax3.set_xlabel("Dimension 1 weights", color="r")
    texts = []
    for i in range(len(dim1_weights)):
        ax3.arrow(
            0,
            0,
            dim1_weights[i],
            dim2_weights[i],
            head_width=0.01,
            head_length=0.01,
            fc="r",
            ec="r",
            alpha=0.6,
        )
        if labels is not None:
            texts.append(
                ax3.text(
                    dim1_weights[i] * (padding - 0.1),
                    dim2_weights[i] * (padding - 0.1),
                    labels[i],
                    color="r",
                    ha="center",
                    va="center",
                )
            )
    adjust_text(texts)
    sns.despine(left=True, right=True, top=True, bottom=True)

    for cax in [ax, ax2, ax3]:
        cax.tick_params(axis="both", which="both", length=0)
    if display:
        plt.show()
    return fig, (ax, ax2, ax3)


def biplot_paired(
    x_components: np.ndarray,
    y_components: np.ndarray,
    x_weights: np.ndarray,
    y_weights: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    n_comps: int,
    hue_x: List[str] = None,
    hue_y: List[str] = None,
    palette: str = "plasma",
    padding: float = 1.2,
    plot_weights: bool = True,
):
    """

    Args:
        x_components:
        y_components:
        x_weights:
        y_weights:
        x_labels:
        y_labels:
        n_comps:
        hue_x:
        hue_y:
        palette:
        padding:
        plot_weights:
        display:

    Returns:

    """
    fig, axes = plt.subplots(
        nrows=n_comps, ncols=n_comps, figsize=(n_comps * 7, n_comps * 7)
    )
    for row_idx in range(n_comps):
        for col_idx in range(n_comps):
            if col_idx < row_idx:  # plot x components
                fig, (ax, ax2, ax3) = biplot_decompose(
                    x_components[:, col_idx],
                    x_components[:, row_idx],
                    x_weights[:, col_idx],
                    x_weights[:, row_idx],
                    x_labels,
                    hue=hue_x,
                    palette=palette,
                    ax=axes[row_idx, col_idx],
                    padding=padding,
                    plot_weights=plot_weights,
                    display=False,
                )
                ax.set_xlabel(f"X Dimension {col_idx+1}", color="b")
                ax.set_ylabel(f"X Dimension {row_idx+1}", color="b")
                ax2.set_ylabel(f"X Dimension {row_idx+1} weights", color="r")
                ax3.set_xlabel(f"X Dimension {col_idx+1} weights", color="r")
            if row_idx == col_idx:  # plot correlations
                cax = axes[row_idx, col_idx]
                if hue_x is None:
                    if hue_y is None:
                        sns.scatterplot(
                            x=x_components[:, row_idx],
                            y=y_components[:, col_idx],
                            ax=cax,
                            color="b",
                        )
                    else:
                        sns.scatterplot(
                            x=x_components[:, row_idx],
                            y=y_components[:, col_idx],
                            ax=cax,
                            hue=hue_y,
                            palette=palette,
                        )
                else:
                    sns.scatterplot(
                        x=x_components[:, row_idx],
                        y=y_components[:, col_idx],
                        ax=cax,
                        hue=hue_x,
                        palette=palette,
                    )

                cax.set_xlabel(f"X Dimension {col_idx+1}")
                cax.set_ylabel(f"Y Dimension {row_idx+1}")
            if col_idx > row_idx:  # plot y components
                fig, (ax, ax2, ax3) = biplot_decompose(
                    y_components[:, col_idx],
                    y_components[:, row_idx],
                    y_weights[:, col_idx],
                    y_weights[:, row_idx],
                    y_labels,
                    hue=hue_y,
                    palette=palette,
                    ax=axes[row_idx, col_idx],
                    padding=padding,
                    plot_weights=plot_weights,
                    display=False,
                )
                ax.set_xlabel(f"Y Dimension {col_idx+1}", color="b")
                ax.set_ylabel(f"Y Dimension {row_idx+1}", color="b")
                ax2.set_ylabel(f"Y Dimension {row_idx+1} weights", color="r")
                ax3.set_xlabel(f"Y Dimension {col_idx+1} weights", color="r")
    plt.tight_layout()
    return fig, axes
