#  ZONE T="0.0132334" N=518927 E=1034734 F=FEPOINT ET=TRIANGLE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_pickle('res/plt01929_iso_CRange.pkl')
data = data.sample(10000)
print(data)

k = data['MeanCurvature_progress_variable']

T = data['temp']
Z = data['mixture_fraction']
Y_OH = data['Y_OH']
print(f'min OH: {min(Y_OH)}, max OH: {max(Y_OH)}')

data_for_comparison = [T, Z, Y_OH]
labels_for_comparison = ['Temperature T',
                         'Mixture Fraction Z',
                         'Mass Fraction OH Y(OH)']


def main():
    # plot3D(data)
    plot2D(data)


def plot3D(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(k, T, data['C_H2'], s=0.25)
    ax.set_xlabel('Mean Curvature k')
    ax.set_ylabel('Temperature T')
    ax.set_zlabel('Progress Variable C')
    plt.show()


def plot2D(data):
    C_Range = np.arange(0.05, 1.0, 0.05)
    fig, ax = plt.subplots(1, 1)
    C_Split = data.groupby('C_H2')
    C_Groups = [C_Split.get_group(x) for x in C_Range]
    avg_k = get_avg('MeanCurvature_progress_variable', C_Groups)
    avg_T = get_avg('temp', C_Groups)
    # ax.scatter(C_Range, avg_k)
    ax.scatter(C_Range, avg_T)
    ax.set_xlabel('Progress Variable C')
    # ax.set_ylabel('Average Curvature k')
    ax.set_ylabel('Average Temperature T')
    fig.tight_layout()
    plt.show()


def get_avg(variable, data):
    avg = [sum(x[variable]) / len(x) for x in data]
    return avg



def plot_against_k_2D(cur_data, cur_ax, cur_label, k):
    cur_ax.scatter(k, cur_data, s=0.5)
    cur_ax.set_xlabel('Mean Curvature k')
    cur_ax.set_ylabel(cur_label)
    return


if __name__ == "__main__":
    main()
