import pandas as pd
import numpy as np
from ProbDistFunc import *
import itertools

# AFT=1428.5 K


def main():
    path_data = pd.read_pickle('res/FlamePaths.pkl')
    high_flux = path_data[path_data['flux'] == 'hf']
    med_flux = path_data[path_data['flux'] == 'mf']
    low_flux = path_data[path_data['flux'] == 'lf']

    prog = path_data['prog'].values
    temp = path_data['temp'].values
    mix = path_data['mix'].values

    print(f'high flux:\n{high_flux}')
    print(f'med flux:\n{med_flux}')
    print(f'low flux:\n{low_flux}')

    fig= plt.figure()
    ax = fig.add_subplot(projection='scatter_density')
    # ax = fig.add_subplot()

    plot_against_2D(fig, ax, prog, temp, mix, 'progress variable c', 'temperature T (k)', 'mixture fraction Z')

    # ax = fig.add_subplot(1, 1, 1)
    ax.axhline(y=1428.5, color='r', linestyle='-')
    plt.show()
    return


if __name__ == '__main__':
    main()