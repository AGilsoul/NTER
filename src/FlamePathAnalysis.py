import pandas as pd
import numpy as np
from ProbDistFunc import *
import itertools

# AFT=1428.5 K

def main():
    path_data = pd.read_pickle('res/FlamePaths.pkl')
    high_flux = [col for col in path_data if col.startswith('hf')]
    med_flux = [col for col in path_data if col.startswith('mf')]
    low_flux = [col for col in path_data if col.startswith('lf')]

    prog = list(itertools.chain.from_iterable([path_data[col].values for col in path_data if col.endswith('prog')]))
    temp = list(itertools.chain.from_iterable([path_data[col].values for col in path_data if col.endswith('temp')]))
    mix = list(itertools.chain.from_iterable([path_data[col].values for col in path_data if col.endswith('mix')]))

    print(f'high flux:\n{high_flux}')
    print(f'med flux:\n{med_flux}')
    print(f'low flux:\n{low_flux}')
    fig = plt.figure()
    plot_against_2D(fig, 1, 1, 1, prog, temp, mix, 'progress variable c', 'temperature T (k)', 'mixture fraction Z')

    # ax = fig.add_subplot(1, 1, 1)
    # ax.axhline(y=1428.5, color='r', linestyle='-')
    plt.show()
    return


if __name__ == '__main__':
    main()