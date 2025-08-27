import multiprocessing
import random

import scipy.stats

from formatDAT import formatDAT
from ReadDat import ReadDat
from ReadCsv import ReadCsv
import numpy as np
from csvToPkl import CSVtoPKL
from scipy.spatial import Delaunay, ConvexHull, KDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as mtri
from ProbDistFunc import pdf_2D, plot_against_2D
import statistics
from pathlib import Path
from formatCSV import formatCSV
import pandas as pd
from copy import copy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Point:
    n = None

    def __init__(self, point_data, norm=False):
        self.X = copy(point_data['X'])
        self.U = copy(point_data['U'])
        self.K = copy(point_data['K'])
        self.T = copy(point_data['T'])
        self.t = copy(point_data['t'])
        if not norm:
            self.compute_norm(point_data['grad'])
        else:
            self.n = copy(point_data['n'])

    def compute_norm(self, grad):
        magnitude = np.sqrt(sum([i**2 for i in grad]))
        self.n = [i/magnitude for i in grad]


class Simplex:
    def __init__(self, indices, points):
        self.v = [indices[0], indices[1], indices[2]]
        self.avg_X, self.avg_U, self.avg_n, self.avg_K, self.avg_T = [], [], [], 0, 0
        self.compute_avgs(points)

    def compute_avgs(self, points):
        self.avg_K = sum(points[v].K for v in self.v)/3
        self.avg_T = sum(points[v].T for v in self.v)/3
        for i in range(3):
            self.avg_X.append(sum([points[v].X[i] for v in self.v])/3)
            self.avg_U.append(sum([points[v].U[i] for v in self.v])/3)
            self.avg_n.append(sum([points[v].n[i] for v in self.v])/3)
        # print(f'avg X: {self.avg_X}')

    def interp_values(self, coords, points):
        cur_points = [points[self.v[i]] for i in range(3)]
        diffs = [[cur_points[i].X[u] - coords[u] for i in range(3)] for u in range(3)]
        dists = [np.dot(diffs[i], diffs[i])**0.5 for i in range(3)]

        total_dist = sum(dists)
        unit_dists = dists / total_dist

        X = [sum([unit_dists[i] * cur_points[i].X[u] for i in range(3)]) for u in range(3)]
        U = [sum([unit_dists[i] * cur_points[i].U[u] for i in range(3)]) for u in range(3)]
        n = [sum([unit_dists[i] * cur_points[i].n[u] for i in range(3)]) for u in range(3)]
        K = sum([unit_dists[i] * cur_points[i].K for i in range(3)])
        T = sum([unit_dists[i] * cur_points[i].T for i in range(3)])
        return {'Success': 1.0,
                'X': X,
                'U': U,
                'n': n,
                'K': K,
                'T': T}


class IsoSurface:
    iso_tol = 0.01
    def_threshold = 0.00025
    data_threshold = 10
    points, simplices = None, None

    def __init__(self, file_name, time=None, p0=None, xlim=[0, 0.032], ylim=[0, 0.032]):
        print(f'Constructing IsoSurface from {file_name} ...')
        if time is None:
            formatDAT(file_name)
        self.t, self.data = ReadDat(file_name)
        self.data = self.data[(self.data['X'] < xlim[1]) & (self.data['X'] > xlim[0]) & (self.data['Y'] < ylim[1]) & (self.data['Y'] > ylim[0])]
        self.data_temp = self.data.copy()
        print(f'Read {len(self.data)} rows of data ...')
        if p0 is None:
            self.make_points()
            self.triangulation()
        else:
            self.trim_data(p0, self.def_threshold)
        self.num_pts = len(self.points)
        self.num_simplices = len(self.simplices)

    def make_points(self):
        self.points = [Point({'X': [self.data.iloc[i]['X'], self.data.iloc[i]['Y'], self.data.iloc[i]['Z']],
                              'U': [self.data.iloc[i]['x_velocity'], self.data.iloc[i]['y_velocity'], self.data.iloc[i]['z_velocity']],
                              'grad': [self.data.iloc[i]['progress_variable_gx'], self.data.iloc[i]['progress_variable_gy'], self.data.iloc[i]['progress_variable_gz']],
                              'K': self.data.iloc[i]['MeanCurvature_progress_variable'],
                              'T': self.data.iloc[i]['temp']}) for i in range(len(self.data))]

    def trim_data(self, X, threshold):
        self.data['dist'] = ((self.data['X'] - X[0])**2 + (self.data['Y'] - X[1])**2 + (self.data['Z'] - X[2])**2)**0.5
        self.data = self.data[self.data['dist'] < threshold]
        if len(self.data) < self.data_threshold:
            return False
        self.make_points()
        self.triangulation()
        self.num_pts = len(self.points)
        self.num_simplices = len(self.simplices)
        return True

    def reset_point(self, p: Point, threshold=def_threshold):
        self.data = self.data_temp.copy()
        return self.trim_data(p.X, threshold=threshold)

    def triangulation(self):
        x = self.data['X'].values
        y = self.data['Y'].values
        X = np.array([x, y]).T
        tri = Delaunay(X)
        simp_list = tri.simplices
        self.simplices = [Simplex(i, self.points) for i in simp_list]


class ParallelIsoSurface:
    iso_tol = 0.01
    def_threshold = 0.00005

    @staticmethod
    def make_points(data, time):
        return [Point({'X': [data.iloc[i]['X'], data.iloc[i]['Y'], data.iloc[i]['Z']],
                       'U': [data.iloc[i]['x_velocity'], data.iloc[i]['y_velocity'], data.iloc[i]['z_velocity']],
                       'grad': [data.iloc[i]['progress_variable_gx'], data.iloc[i]['progress_variable_gy'], data.iloc[i]['progress_variable_gz']],
                       'K': data.iloc[i]['MeanCurvature_progress_variable'],
                       'T': data.iloc[i]['temp'],
                       't': time}) for i in range(len(data))]

    @staticmethod
    def trim_data(data, X, threshold):
        data['dist'] = ((data['X'] - X[0])**2 + (data['Y'] - X[1])**2 + (data['Z'] - X[2])**2)**0.5
        return data[data['dist'] < threshold]

    @staticmethod
    def triangulation(data, points):
        x = data['X'].values
        y = data['Y'].values
        X = np.array([x, y]).T
        tri = Delaunay(X)
        simp_list = tri.simplices
        all_simplicies = [Simplex(i, points) for i in simp_list]
        # print(f'x coords: {[all_simplicies[i].avg_X[0] for i in range(len(all_simplicies))]}')
        # plot_triangulated_surface([x, y, data['Z']], data.iloc[0]["t"], all_simplicies, simp_list)

        return all_simplicies

    @staticmethod
    def get_trimmed_data(data, point, time):
        trimmed_df = ParallelIsoSurface.trim_data(data, copy(point.X), ParallelIsoSurface.def_threshold)
        points = ParallelIsoSurface.make_points(trimmed_df, time)
        simplices = ParallelIsoSurface.triangulation(trimmed_df, points)
        return points, simplices

    @staticmethod
    def get_all_data(data, time, xlim=[0, 0.032], ylim=[0, 0.032]):
        data = data[(data['X'] < xlim[1]) & (data['X'] > xlim[0]) & (data['Y'] < ylim[1]) & (data['Y'] > ylim[0])]
        points = ParallelIsoSurface.make_points(data, time)
        simplices = ParallelIsoSurface.triangulation(data, points)
        return points, simplices

    @staticmethod
    def get_all_points(data, time, xlim=[0, 0.032], ylim=[0, 0.032]):
        data = data[(data['X'] < xlim[1]) & (data['X'] > xlim[0]) & (data['Y'] < ylim[1]) & (data['Y'] > ylim[0])]
        points = ParallelIsoSurface.make_points(data, time)
        return points


def plot_triangulated_surface(X, t, simplicies, simp_list):
    x, y, z = X[0], X[1], X[2]
    curvatures = np.array([simplicies[i].avg_X[2] for i in range(len(simplicies))])
    # print(curvatures)
    my_cmap = plt.get_cmap('hot')
    norm = mpl.colors.Normalize(vmin=min(curvatures), vmax=max(curvatures))
    color_list = my_cmap(norm(curvatures))
    # print(f'color list: {color_list}')
    # new_X = X

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(x, y, z, triangles=simp_list)
    ax.set_title(f'timestep: {t}')
    surf.set_fc(color_list)
    plt.show()


def getDelta(point: Point, simplices, points, dt): # simplices_0, points_0, dt):
    threshold = 0.0005
    inter_simplex = None
    minDist = 0
    minQ = None
    t_min = 0
    x_final = 0
    # for every simplex
    for simplex in simplices:
        # get vertex points
        v0, v1, v2 = points[simplex.v[0]], points[simplex.v[1]], points[simplex.v[2]]
        # check to make sure target simplex isn't too far from point
        dist = np.sqrt(sum([(point.X[i] - simplex.avg_X[i])**2 for i in range(3)]))
        if dist > threshold:
            continue

        if abs(v1.X[2] - v0.X[2]) > threshold or abs(v2.X[2] - v0.X[2]) > threshold or abs(v1.X[2] - v2.X[2]) > threshold:
            continue
        # get base vertex and vectors between vertices
        A = v0.X
        AC = [v1.X[i] - v0.X[i] for i in range(3)]
        AB = [v2.X[i] - v0.X[i] for i in range(3)]
        BC = [v1.X[i] - v2.X[i] for i in range(3)]

        AC_len = np.sqrt(sum(x**2 for x in AC))
        AB_len = np.sqrt(sum(x ** 2 for x in AB))
        BC_len = np.sqrt(sum(x ** 2 for x in BC))

        if AC_len < 0.5 * AB_len or AC_len < 0.5 * BC_len or AB_len < 0.5 * AC_len or AB_len < 0.5 * BC_len or BC_len < 0.5 * AC_len or BC_len < 0.5 * AB_len:
            continue

        # calculate normal vector of plane
        plane_norm = np.cross(AB, AC)
        magnitude = np.sqrt(sum(x**2 for x in plane_norm))
        for i in range(3):
            plane_norm[i] /= magnitude

        # make sure vector isn't parallel to simplex
        dot_prod = np.dot(point.n, plane_norm)
        if dot_prod == 0:
            continue

        # get new X by accounting for flow
        new_X = [point.X[i] + point.U[i] * dt for i in range(3)]

        # get intersection point on target plane
        t = (np.dot(plane_norm, A) - np.dot(plane_norm, new_X)) / np.dot(plane_norm, point.n)
        Q = [new_X[i] + point.n[i] * t for i in range(3)]

        crossABQ = np.cross(AB, Q)
        crossACQ = np.cross(AC, Q)
        crossBCQ = np.cross(BC, Q)

        dotAB = np.dot(crossABQ, plane_norm)
        dotAC = np.dot(crossACQ, plane_norm)
        dotBC = np.dot(crossBCQ, plane_norm)

        in_bounds = dotAB > 0 and dotBC > 0 and dotAC > 0
        if in_bounds:
            if inter_simplex is None:
                inter_simplex = simplex
                minDist = dist
            elif dist < minDist:
                inter_simplex = simplex
                minDist = dist
            minQ = Q
            t_min = t
            x_final = new_X

    if inter_simplex is None:
        return {'Success': 0.0}, 0


    new_X_tri = np.array([[points[i].X[0], points[i].X[1], points[i].X[2]] for i in range(len(points))]).T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(new_X_tri[0], new_X_tri[1], new_X_tri[2], triangles=[simplices[i].v for i in range(len(simplices))], alpha=0.5)
    ax.quiver(x_final[0], x_final[1], x_final[2], point.n[0] * t_min, point.n[1] * t_min, point.n[2] * t_min)
    plt.show()


    return inter_simplex.interp_values(minQ, points), t_min


def delta_point(point: Point, simplices, points, dt, index): # simplices_0, points_0, dt, index):
    del_res, t_min = getDelta(point, simplices, points, dt) # simplices_0, points_0, dt)
    if not del_res['Success']:
        return del_res
    deltaK = del_res['K'] - point.K
    delta_r = (1 / (point.K + deltaK)) - (1 / point.K)

    drdt = delta_r / dt
    kdrdt = point.K * drdt

    if np.isnan(delta_r) or np.isinf(delta_r):
        return {'Success': 0.0}
    # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
    return {'Success': 1.0,
            'dK': deltaK,
            'dr': delta_r,
            'K': point.K,
            'r': 1/point.K,
            'temp': point.T,
            'drdt': drdt,
            'kdrdt': kdrdt,
            'dt': dt,
            'X0': point.X,
            'X1': del_res['X'],
            'U1': del_res['U'],
            'n1': del_res['n'],
            'T1': del_res['T'],
            'K1': del_res['K'],
            't': point.t + dt,
            'index': index,
            't_min': t_min
            }


def main():
    getPointPath()
    return


def plotDataPDF(data, x_label, y_label, log=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    if log:
        ys = []
        for y in data[y_label]:
            if y > 0:
                ys.append(np.log(y))
            elif y < 0:
                ys.append(-np.log(abs(y)))
            else:
                ys.append(0)
        pdf_2D(fig, ax, data[x_label], ys, x_label, y_label, dpi=75)
    else:
        pdf_2D(fig, ax, data[x_label], data[y_label], x_label, y_label, dpi=75)
    plt.show()


def plotDataHeatMap(data, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_against_2D(fig, ax, data[x_label], data[y_label], data[z_label], x_label, y_label, z_label, bins=[200, 200])
    plt.show()


def plotPointTrajectories(data):
    data = data[abs(data['dr']) < 0.1]
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('start_data')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('kdrdt_data')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('dr_data')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('surface_data')
    # ax6 = fig.add_subplot(2, 3, 6, projection='scatter_density')
    # ax6.set_title('kdrdt_data')
    bins = [500, 500]
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.01, 0.02], ylim=[0.01, 0.02])
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT0 = IsoSurface('res/testIso01920.dat')
    # plot_against_2D(fig, ax1,
    #                 data['x0'], data['y0'], data['z0'],
    #                 'x', 'y', 'z', bins=bins, scale='log')
    plot_against_2D(fig, ax1,
                    data['X0'][0], data['X0'][1], data['X0'][2],
                    'x', 'y', 'z', bins=bins, scale='log')
    plot_against_2D(fig, ax2,
                    data['X0'][0], data['X0'][1], abs(data['kdrdt']),
                    'x', 'y', 'kdrdt', bins=bins, scale='log')
    plot_against_2D(fig, ax3,
                    data['X0'][0], data['X0'][1], abs(data['dr']),
                    'x', 'y', 'dr', bins=bins, scale='log')
    # plot_against_2D(fig, ax4,
                    # data['x1'], data['y1'], abs(data['K']),
                    # 'x', 'y', 'k', bins=bins, scale='log')
    # plot_against_2D(fig, ax4,
    #                 isoT0.data['X'], isoT0.data['Y'], isoT0.data['Z'],
    #                 'x', 'y', 'z', bins=bins, scale='log')

    # pdf_2D(fig, ax6, data['K'], data['kdrdt'], 'K', 'kdrdt')

    '''
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    print(isoT1.tri.simplices)
    print(isoT1.X[0])
    print(isoT1.X[1])
    print(isoT1.X[2])
    
    ax.plot_trisurf(isoT1.X.T[0], isoT1.X.T[1], isoT1.X.T[2], triangles=isoT1.tri.simplices, cmap=plt.cm.Spectral)
    data = data.sample(n=10000)
    for i in range(len(data)):
        row = data.iloc[i]
        ax.plot([row['x0'], row['x1']], [row['y0'], row['y1']], [row['z0'], row['z1']], 'ro-', markersize=0.1)
        # ax.scatter([row['x0']], [row['y0']], [row['z0']], s=0.1, color='blue')
        '''
    plt.tight_layout()
    plt.show()
    return


def async_loop(index, point: Point, isoT1: pd.DataFrame, dt):
    if index % 1000 == 0:
        print(f'Running calculations for point {index} ...')
    points_1, simplices_1 = ParallelIsoSurface.get_trimmed_data(isoT1, point, point.t + dt)
    delta = delta_point(point, simplices_1, points_1, dt, index)
    if index % 100 == 0:
        print(f'Finished calculations for point {index}.')
    return delta


def getPointPath():
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.01, 0.015], ylim=[0.01, 0.015])
    # isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.01, 0.015], ylim=[0.01, 0.015])
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT0 = IsoSurface('res/testIso01920.dat')
    # isoT1 = IsoSurface('res/testIso01930.dat')
    print('Formatting DAT files ...')
    formatDAT('res/testIso03850.dat')
    formatDAT('res/testIso03851.dat')
    print('Reading DAT files ...')
    t0, isoT0 = ReadDat('res/testIso03850.dat')

    t1, isoT1 = ReadDat('res/testIso03851.dat')
    dt = t1 - t0
    print('Getting point data ...')
    # xlim, ylim = [0.01, 0.015], [0.01, 0.015]
    xlim, ylim = [0.0, 0.032], [0.0, 0.032]
    points = ParallelIsoSurface.get_all_points(isoT0, t0, xlim=xlim, ylim=ylim)
    print('Finished constructing isosurfaces!')
    isoT1 = isoT1[(isoT1['X'] < xlim[1]) & (isoT1['X'] > xlim[0]) & (isoT1['Y'] < ylim[1]) & (isoT1['Y'] > ylim[0])]
    print('Making mapping list ...')
    delta_list = [[i, points[i], isoT1, dt] for i in range(len(points))]
    res_list = []
    print('Beginning multiprocessing ...')
    with multiprocessing.Pool() as pool:
        res_list = pool.starmap_async(async_loop, delta_list).get()
    print('Finished multiprocessing ...')
    print(res_list)
    out_df = pd.DataFrame.from_dict(res_list)
    out_df = out_df[out_df['Success'] != 0.0]
    print(out_df)
    out_df.to_pickle('res/TemporalPathData_all.pkl')
    print('Done!')
    plotDataPDF(out_df, 'K', 'kdrdt')
    return


def readPkl(file_name):
    # file_name = 'res/TemporalPathData_all.pkl'
    data = pd.read_pickle(file_name)
    print(data)
    print(f'read {len(data)} rows of data')
    # print(f'r:\n{data["r"]}')
    # print(f'max r: {max(data["r"])}')
    # print(f'min r: {min(data["r"])}')
    # print(f'dr:\n{data["dr"]}')
    # print(f'max dr: {max(data["dr"])}')
    # print(f'min dr: {min(data["dr"])}')
    # print(data.columns)
    print(data['MeanCurvature_progress_variable'])
    plotDataHeatMap(data, 'X', 'Y', 'MeanCurvature_progress_variable')
    # plotPointTrajectories(data)
    # plotDataPDF(data, 'r', 'dr')
    # data2 = pd.read_pickle('res/TemporalPathData_fourth_last_test.pkl')
    # data2 = data2[(data2['x0'] < 0.015) & (data2['x0'] > 0.01) & (data2['y0'] < 0.015) & (data2['y0'] > 0.01)]
    # plotDataPDF(data, 'dK', 'kdrdt', log=True)

    # print(data.iloc[0]['X1'])
    # print(data.iloc[1]['X1'])
    # print(data.iloc[2]['X1'])

    # plot_point_paths(data, 't', 'kdrdt')
    # plotDataHeatMap(data, 'K', 'kdrdt', 'temp')


def get_time_file(timestep):
    file_name = f'res/SeqPaths/subPathIso0{timestep}.pkl'
    return pd.read_pickle(file_name)


def get_reg_file(timestep):
    file_name = f'res/SeqPaths/regSubPathIso0{timestep}.pkl'
    return pd.read_pickle(file_name)


def plot_point_paths(data, x_label, y_label, log=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    num_indices = len(pd.unique(data['index']))
    for i in range(num_indices):
        cur_data = data[data['index'] == i]
        x_data = cur_data[x_label].values[1:-20]
        y_data = cur_data[y_label].values[1:-20]
        print(x_data)
        ax.plot(x_data, y_data)
    plt.show()


def regular_grid(data, i, grid_dim=450):
    cols = ['X', 'Y', 'Z', 'x_velocity', 'y_velocity', 'z_velocity', 'density', 'temp', 'HeatRelease', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_H', 'Y_O', 'Y_OH', 'Y_HO2', 'Y_H2O2', 'Y_N2', 'mixture_fraction', 'progress_variable', 'progress_variable_gx', 'progress_variable_gy', 'progress_variable_gz', '||gradprogress_variable||', 'MeanCurvature_progress_variable', 'StrainRate_progress_variable', 't']
    print(f'making regular surface from {len(data)} points at timestep {i}...')
    points = np.array([data['X'].values, data['Y'].values]).T
    max_x = max(data['X'].values)
    max_y = max(data['Y'].values)
    step_size_x = max_x / grid_dim
    step_size_y = max_y / grid_dim
    tree = KDTree(points, leafsize=200000)
    all_data = []
    k = 3
    for x in range(grid_dim):
        for y in range(grid_dim):
            x_coord = (x + 0.5) * step_size_x
            y_coord = (y + 0.5) * step_size_y
            nearest_points = tree.query([x_coord, y_coord], k=k)
            pt_data = [x_coord, y_coord]
            for i in range(2, len(cols) - 1):
                pt_data.append(sum(data.iloc[nearest_points[1]][cols[i]])/k)
            pt_data.append(data.iloc[0]['t'])
            all_data.append(pt_data)

    new_data = pd.DataFrame(all_data)
    new_data.columns = cols
    # print(new_data.columns)
    # print(new_data['MeanCurvature_progress_variable'])
    # print('Done!')
    return new_data


def find_nearest_point(data, point):
    points = np.array([data['X'].values, data['Y'].values]).T
    tree = KDTree(points, leafsize=2000)
    nearest_points = tree.query([point[0], point[1]], k=1)
    return nearest_points[1]


# FIX POINT DATA CHANGING
def singleTemporalPath():
    time_range = range(3851, 3887, 1)
    small_range = range(3852, 3887, 1)
    print(f'Getting initial isosurface file ...')
    isoT0 = get_reg_file(time_range[0])
    # isoT0 = get_time_file(time_range[0])
    # test_iso = get_time_file(time_range[0])
    # isoT0 = isoT0.sample(frac=0.5)
    # isoT0 = regular_grid(isoT0)
    t0 = isoT0.iloc[0]['t']
    # all_points, all_simplicies = ParallelIsoSurface.get_all_data(isoT0, t0)
    print(f'Getting initial isosurface points ...')
    all_points, all_simplices = ParallelIsoSurface.get_all_data(isoT0, t0)
    num_rand = 1
    point_index = 6000
    last_points = all_points
    last_simplices = all_simplices
    point_data = []
    index = 0
    # for index in range(num_rand):
    # print(f'Getting path of point {index} ...')
    # point_index = random.randint(0, len(all_points))
    # point_index = 169000
    # point_index = 150455
    point_0 = all_points[point_index]
    # print(f'X: {point_0.X}')
    # print(find_nearest_point(test_iso, point_0.X))
    # last_points = all_points
    # last_simplices = all_simplicies

    for step in range(len(small_range)):
        isoT1 = get_reg_file(small_range[step])
        print(f'Getting next isosurface file ...')
        # isoT1 = get_time_file(small_range[step])
        # isoT1 = isoT1.sample(frac=0.5)
        # isoT1 = regular_grid(isoT1)
        t1 = isoT1.iloc[0]['t']
        print(f'path from {t0} to {t1} ...')
        dt = t1 - t0
        print(f'Getting next isosurface points and simplices ...')
        points_1, simplices_1 = ParallelIsoSurface.get_trimmed_data(copy(isoT1), copy(point_0), copy(t1))
        print(f'Calculating path between surfaces ...')
        delta_res = delta_point(point_0, simplices_1, points_1, dt, index)  # last_simplices, last_points, dt, index)
        if step != 0:
            print(f'plotting ...')

            new_X_tri = np.array([[points_1[i].X[0], points_1[i].X[1], points_1[i].X[2]] for i in range(len(points_1))]).T
            new_X_tri_0 = np.array(
                [[last_points[i].X[0], last_points[i].X[1], last_points[i].X[2]] for i in range(len(last_points))]).T

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_trisurf(new_X_tri[0], new_X_tri[1], new_X_tri[2],
                            triangles=[simplices_1[i].v for i in range(len(simplices_1))], alpha=0.5)
            ax.plot_trisurf(new_X_tri_0[0], new_X_tri_0[1], new_X_tri_0[2],
                            triangles=[last_simplices[i].v for i in range(len(last_simplices))], alpha=0.5)
            ax.quiver(delta_res['X1'][0], delta_res['X1'][1], delta_res['X1'][2], point_0.n[0] * delta_res['t_min'], point_0.n[1] * delta_res['t_min'], point_0.n[2] * delta_res['t_min'])

            plt.show()

        if delta_res['Success'] == 0.0:
            print(f'Unsuccessful for point {index}')
            break
        point_data.append(delta_res)
        point_0 = copy(Point({'X': delta_res['X1'],
                              'U': delta_res['U1'],
                              'n': delta_res['n1'],
                              'K': delta_res['K1'],
                              'T': delta_res['T1'],
                              't': t1}, norm=True))
        t0 = t1
        last_points = points_1
        last_simplices = simplices_1

    out_df = pd.DataFrame.from_dict(point_data)
    out_df = out_df[out_df['Success'] != 0.0]
    print(out_df)
    out_df.to_pickle('res/TemporalPathData_all.pkl')
    print('Done!')
    plot_point_paths(out_df, 't', 'kdrdt')
    # plotDataPDF(out_df, 't', 'K')


if __name__ == '__main__':
    # main()
    # readPkl('res/SeqPaths/regSubPathIso03850.pkl')
    # readPkl('res/SeqPaths/subPathIso03850.pkl')
    singleTemporalPath()
    # time_range = range(3851, 3887, 1)
    # for i in range(len(time_range)):
        # res = regular_grid(get_time_file(time_range[i]), time_range[i])
        # pd.to_pickle(res, f'res/SeqPaths/regSubPathIso0{time_range[i]}.pkl')


