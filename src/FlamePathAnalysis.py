import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import atan2, degrees
from matplotlib.collections import LineCollection
from sklearn import preprocessing as pre
from ProbDistFunc import *
import itertools
from Cantera import *

# AFT=1428.5 K


def main():
    path_data = pd.read_pickle('res/FlamePaths.pkl')
    # print(path_data['flux'])

    hf = path_data[path_data['flux'] == 'hf']
    mf = path_data[path_data['flux'] == 'mf']
    print(f'final mf prog: {max(mf["prog"])}')
    print(f'final mf coords: {mf["path"].values[-1]}')
    lf = path_data[path_data['flux'] == 'lf']

    # print(hf)
    # print(lf)

    # print(f'max r: {max(path_data["r"].values)}')
    # print(f'max data points: {max([len(hf["r"]), len(mf["r"]), len(lf["r"])])}')
    prog = list(path_data['prog'].values)
    temp = list(path_data['temp'].values)
    mix = list(path_data['mix'].values)
    # print(f'num prog vars: {len(prog)}')

    fig = plt.figure(figsize=(15, 15))
    matplotlib.rcParams.update({'font.size': 22})
    # ax = fig.add_subplot(projection='scatter_density')
    ax = fig.add_subplot(1, 1, 1)
    # ax1 = fig.add_subplot(2, 1, 2)
    cantera_temp, cantera_c, cantera_z, grid = compute1DFlame()
    prog.extend(cantera_c)
    temp.extend(cantera_temp)
    mix.extend(cantera_z)

    rs = [hf['r'].values, mf['r'].values, lf['r'].values]
    progs = [hf['prog'].values, mf['prog'].values, lf['prog'].values]
    temps = [hf['temp'].values, mf['temp'].values, lf['temp'].values]
    mixes = [hf['mix'].values, mf['mix'].values, lf['mix'].values]
    labels = ['high flux', 'medium flux', 'low flux']

    print(f'max hf temp: {max(hf["temp"].values)}')
    print(f'max mf temp: {max(mf["temp"].values)}')
    print(f'max lf temp: {max(lf["temp"].values)}')

    for i in range(len(progs)):
        points = np.array([progs[i], temps[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(mix), max(mix))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(mixes[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        # ax1.plot(x_norm, temps[i], label=labels[i])
    # ax1.axhline(y=1428.5, color='black', label='AFT', linestyle='dashed')
    fig.colorbar(line, ax=ax, label='Mixture Fraction Z')
    ax.plot(cantera_c, cantera_temp, linewidth=2, color='red', label='Cantera 1D', linestyle='dashed')

    cantera_r = np.array(grid).reshape(-1, 1)
    cantera_r_norm = pre.MinMaxScaler().fit_transform(cantera_r)
    # ax1.plot(cantera_r_norm, cantera_temp, linewidth=1, color='red', label='Cantera 1D', linestyle='dashed')
    ax.axhline(y=1428.5, color='black', label='AFT', linestyle='dashed')
    ax.set_xlim(0, 1)
    ax.set_ylim(min(temp), 1600)
    # ax1.set_ylim(min(temp), 1600)
    ax.set_xlabel('Progress Variable c')
    ax.set_ylabel('Temperature T (K)')
    # ax1.set_xlabel('normalized r')
    # ax1.set_ylabel('temp T (K)')

    plt.legend(loc='upper left')
    plt.show()
    return


if __name__ == '__main__':
    main()
