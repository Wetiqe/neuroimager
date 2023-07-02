import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def large_font_size():
    parameters = {
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "figure.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 15,
        "legend.title_fontsize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.titleweight": "bold",
        "font.weight": "bold",
        "font.sans-serif": "Arial",
    }
    plt.rcParams.update(parameters)


def small_font_size():
    parameters = {
        "axes.labelsize": 6,
        "axes.titlesize": 12,
        "figure.titlesize": 12,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
    }
    plt.rcParams.update(parameters)


def no_edge(ax):
    sns.set_style("white")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    return ax
