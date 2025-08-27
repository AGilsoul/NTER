from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import matplotlib
import mpl_scatter_density

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


def plot_against_2D(fig, ax, xd, yd, z, xlabel, ylabel, zlabel, bins=[500, 500], scale=None):
    # z = [d if d >= 0 else 0 for d in z]
    H, x_edges, y_edges, bin_num = binned_statistic_2d(xd, yd, values=z, statistic='mean', bins=bins)
    H = np.ma.masked_invalid(H)
    XX, YY = np.meshgrid(x_edges, y_edges)
    if scale is None:
        p1 = ax.pcolormesh(XX, YY, H.T)
    else:
        p1 = ax.pcolormesh(XX, YY, H.T, norm=matplotlib.colors.LogNorm())
    cbar = fig.colorbar(p1, ax=ax, label=zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


def pdf_2D(fig, ax, xd, yd, xlabel, ylabel, dpi=70, scale=None):
    if scale is not None:
        ax.set_xscale('log')
    density = ax.scatter_density(xd, yd, cmap=white_viridis, dpi=dpi, norm=matplotlib.colors.SymLogNorm(linthresh=0.03))
    fig.colorbar(density, label='Number of points per pixel', vmin=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return
