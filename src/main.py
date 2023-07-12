#  ZONE T="0.0132334" N=518927 E=1034734 F=FEPOINT ET=TRIANGLE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density
from scipy.stats import binned_statistic_2d

prog_var = 'H2O'
data = pd.read_pickle(f'res/plt01929_iso_CRange_Y{prog_var}.pkl').dropna()
# data = data.sample(10000)
data = data[data[f'C_{prog_var}'] == 0.8]

k = data['MeanCurvature_progress_variable']
Z = data['mixture_fraction']
C = data[f'C_{prog_var}']
Y_OH = data['Y_OH']
Y_H = data['Y_H']

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


def main():
    # plot3D(data)
    plot2D(data)


def plot3D(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(k, Z, C, s=0.25)
    ax.set_xlabel('Mean Curvature k')
    ax.set_ylabel('Mixture Fraction Z')
    ax.set_zlabel('Progress Variable C')
    plt.show()


def plot2D(cur_data):
    fig = plt.figure()
    # plot_against_2D(fig, k, C, 'Curvature K', 'Prog Var C(H2)')
    rows = 2
    cols = 2
    plot_against_2D(fig, rows, cols, 1, k, Z, Y_OH, 'Curvature k', 'Mixture Fraction Z', 'Mass Fraction OH')
    plot_against_2D(fig, rows, cols, 2, k, Z, Y_H, 'Curvature k', 'Mixture Fraction Z', 'Mass Fraction H')
    # plot_against_2D(fig, rows, cols, 3, k, Z, Y_OH, 'Curvature k', 'Mixture Fraction Z', 'Mass Fraction OH')
    # plot_against_2D(fig, rows, cols, 4, k, Z, Y_OH, 'Curvature k', 'Mixture Fraction Z', 'Mass Fraction OH')

    # pdf_2D(fig, rows, cols, 2, k, C, 'Curvature k', f'Prog Var C({prog_var})')
    # pdf_2D(fig, rows, cols, 3, C, Z, f'Prog Var C({prog_var})', 'Mixture Fraction Z')
    fig.tight_layout()
    plt.show()


def get_avg(variable, cur_data):
    avg = [sum(x[variable]) / len(x) for x in cur_data]
    return avg


def plot_against_2D(fig, r, c, i, xd, yd, z, xlabel, ylabel, zlabel):
    z = [d if d >= 0 else 0 for d in z]
    ax = fig.add_subplot(r, c, i)
    H, x_edges, y_edges, bin_num = binned_statistic_2d(xd, yd, values=z, statistic='mean', bins=[750, 750])
    H = np.ma.masked_invalid(H)
    XX, YY = np.meshgrid(x_edges, y_edges)
    p1 = ax.pcolormesh(XX, YY, H.T)
    cbar = fig.colorbar(p1, ax=ax, label=zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


def pdf_2D(fig, r, c, i, xd, yd, xlabel, ylabel):
    ax = fig.add_subplot(r, c, i, projection='scatter_density')
    density = ax.scatter_density(xd, yd, cmap=white_viridis, dpi=70)
    fig.colorbar(density, label='Number of points per pixel')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


if __name__ == "__main__":
    main()
