import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def get_sig(sig):
    symbol = ""
    if 0.01 < sig < 0.05:
        symbol = "*"
    elif 0.001 < sig <= 0.01:
        symbol = "**"
    elif sig <= 0.001:
        symbol = "***"
    elif 0.05 <= sig < 0.10:
        symbol = "^"

    return sig, symbol


def density_scatter(
    x, y, reg=True, gaussian_density=True, figsize=(10, 10), display=True, **kwargs
):
    xy = np.vstack([x, y])
    if gaussian_density:
        z = gaussian_kde(xy)(xy)
    else:
        H, x_edges, y_edges = np.histogram2d(x, y)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Convert the histogram to a 2D array of densities
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - y_centers[0]
        z = H / (dx * dy)
    # Create the scatter plot with density-based coloring
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x=x, y=y, c=z, cmap="viridis", **kwargs)
    sns.regplot(x=x, y=y, scatter=False, ax=ax, color="black")

    # Add a colorbar to the figure
    cbar = plt.colorbar(sc)
    cbar.set_label("Density")
    if display:
        plt.show()

    return fig, ax


def plot_time_series(
    time_series,
    n_samples=None,
    y_tick_values=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot a time series with channel separation.
    ***This is a function modified from else where. Add the source***
    Parameters
    ----------
    time_series : numpy.ndarray
        The time series to be plotted. Shape must be (n_samples, n_channels).
    n_samples : int
        The number of time points to be plotted.
    y_tick_values:
        Labels for the channels to be placed on the y-axis.
    fig_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.subplots.
    plot_kwargs : dict
        Keyword arguments to be passed on to matplotlib.pyplot.plot.
    ax : matplotlib.axes.Axes
        The axis on which to plot the data. If not given, a new axis is created.
    filename : str
        Output filename.
    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure object.
    ax : matplotlib.pyplot.axis.
        Matplotlib axis object(s).
    """
    time_series = np.asarray(time_series)
    n_samples = min(n_samples or np.inf, time_series.shape[0])
    n_channels = time_series.shape[1]

    # Validation
    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    default_fig_kwargs = {"figsize": (12, 8)}
    if fig_kwargs is None:
        fig_kwargs = default_fig_kwargs
    else:
        # fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)
        pass
    default_plot_kwargs = {"lw": 0.7, "color": "tab:blue"}
    if plot_kwargs is None:
        plot_kwargs = default_plot_kwargs
    else:
        # plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)
        pass
    # Calculate separation
    separation = (
        np.maximum(time_series[:n_samples].max(), time_series[:n_samples].min()) * 1.2
    )
    gaps = np.arange(n_channels)[::-1] * separation

    # Create figure
    if ax is None:
        fig, ax = plt.figure(**fig_kwargs)

    # Plot data
    ax.plot(time_series[:n_samples] + gaps[None, :], **plot_kwargs)

    ax.autoscale(tight=True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set x and y axis tick labels
    ax.set_xticks([])
    if y_tick_values is not None:
        ax.set_yticks(gaps)
        ax.set_yticklabels(y_tick_values)
    else:
        ax.set_yticks([])

    # Save figure
    # if filename is not None:
    #     save(fig, filename)

    return fig, ax
