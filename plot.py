import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def desity_scatter(x,y, reg=True, gaussian_density=True,figsize=(10,10), display=True, **kwargs):
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
    sns.regplot(x=x, y=y, scatter=False, ax=ax, color='black')

    # Add a colorbar to the figure
    cbar = plt.colorbar(sc)
    cbar.set_label('Density')
    if display:
        plt.show()
    
    return fig, ax
