import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

colours = {
    "dft": "black",
    # empirical colours are blues
    "lcbop": "#6fa8dc",
    "edip": "#77c9a8",
    # synthetic colours are reds and oranges
    # "ace": "#cc4125",
    "ace": "red",
    "gap20": "#ffa319",
    "qsnap": "#f1c232",
}

empirical = ["lcbop", "edip"]
synthetic = ["ace", "gap20", "qsnap"]

style_defaults = {
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # "font.family": "Lato",
    "font.family": "Helvetica",
    "figure.figsize": (4, 4),
    "legend.fancybox": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
}


def log_log(ax=None, yticks=None, xticks=None, dp=1):
    if ax is None:
        ax = plt.gca()
    if yticks is None:
        yticks = [round(y, dp) for y in ax.get_yticks() if y > 0][1:-1]
    if xticks is None:
        xticks = [round(x, dp) for x in ax.get_xticks() if x > 0]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(yticks, yticks)
    ax.set_xticks(xticks, xticks)
    ax.minorticks_off()


def colour_gradient(things, colour):
    cmap = LinearSegmentedColormap.from_list("mycmap", ["white", colour])
    gradient = cmap(np.linspace(0.2, 1, len(things)))

    return zip(things, gradient)
