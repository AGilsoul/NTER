import ctypes
import random
from ctypes import cdll, c_int, c_double, POINTER, c_bool, c_void_p
from formatDAT import formatDAT
from ReadDat import ReadDat
from ReadCsv import ReadCsv
import numpy as np
from csvToPkl import CSVtoPKL
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from ProbDistFunc import pdf_2D, plot_against_2D
import statistics
from pathlib import Path
from formatCSV import formatCSV
import pandas as pd

DoublePtr = POINTER(c_double)
IntPtr = POINTER(c_int)

# lib_path = 'Python/lib/CPUSingleTemporalPath.so'  # linux
lib_path = 'lib/CPUSingleTemporalPath.dll'  # windows
FlamePathLib = cdll.LoadLibrary(lib_path)
getDelta = FlamePathLib.getDelta
getDeltaPoint = FlamePathLib.getResPoint

getDelta.argtypes = [DoublePtr,
                     DoublePtr,
                     DoublePtr,
                     c_double,
                     IntPtr,
                     DoublePtr,
                     DoublePtr,
                     DoublePtr,
                     c_int,
                     c_int,
                     c_double]
getDelta.restype = c_double

getDeltaPoint.argtypes = [DoublePtr,
                     DoublePtr,
                     DoublePtr,
                     c_double,
                     IntPtr,
                     DoublePtr,
                     DoublePtr,
                     DoublePtr,
                     c_int,
                     c_int,
                     c_double]
getDeltaPoint.restype = DoublePtr


class IsoSurface:
    iso_tol = 0.01
    def_threshold = 0.0005
    data_threshold = 10

    def __init__(self, file_name, time=None, iso_val=None, tol=iso_tol, p0=None, xlim=[0, 0.032], ylim=[0, 0.032]):
        print(f'Constructing IsoSurface from {file_name} ...')
        if time is None:
            formatDAT(file_name)
        self.X, self.U, self.flat_X, self.flat_U, self.grad, self.flat_grad, self.simplex_vertices, self.flat_simplices, self.tri = None, None, None, None, None, None, None, None, None
        self.c_simplicies, self.c_X, self.c_U, self.c_K, self.c_T, self.c_r, self.c_grad = None, None, None, None, None, None, None
        if time is not None:
            self.data = pd.read_pickle(file_name).dropna()
            # print(len(self.data))
            self.data = self.data[(self.data['progress_variable'] < (iso_val + tol)) & (self.data['progress_variable'] > (iso_val - tol))]
            # print(len(self.data))
            self.data = self.data.astype(float)
            self.t = time
        else:
            self.t, self.data = ReadDat(file_name)
        self.data = self.data[(self.data['X'] < xlim[1]) & (self.data['X'] > xlim[0]) & (self.data['Y'] < ylim[1]) & (self.data['Y'] > ylim[0])]
        self.data['progress_variable_g_mag'] = (self.data['progress_variable_gx']**2 + self.data['progress_variable_gy']**2 + self.data['progress_variable_gz']**2)**0.5
        self.data_temp = self.data.copy()
        print(f'Read {len(self.data)} rows of data ...')
        print(self.data)
        if p0 is None:
            self.K = np.array(self.data['MeanCurvature_progress_variable'])
            self.r = np.array(abs(1 / self.data['MeanCurvature_progress_variable']))
            self.T = np.array(self.data['temp'])
            self.triangulation()
            self.init_properties()
        else:
            self.trim_data(p0, self.def_threshold)
            self.K = np.array(self.data['MeanCurvature_progress_variable'])
            self.r = np.array(abs(1 / self.data['MeanCurvature_progress_variable']))
            self.T = np.array(self.data['temp'])
        print(self.r)
        self.num_pts = len(self.X)
        self.num_simplices = len(self.simplex_vertices)
        self.get_ctypes()

    def trim_data(self, X, threshold):
        # print(X)
        # print(f'Trimming data, original size: {len(self.data)}')
        self.data['dist'] = ((self.data['X'] - X[0])**2 + (self.data['Y'] - X[1])**2 + (self.data['Z'] - X[2])**2)**0.5
        self.data = self.data[self.data['dist'] < threshold]
        # print(f'Done trimming, new size: {len(self.data)}')
        if len(self.data) < self.data_threshold:
            return False
        self.triangulation()
        self.num_pts = len(self.X)
        self.num_simplices = len(self.simplex_vertices)
        self.init_properties()
        return True

    def reset_point(self, X, threshold=def_threshold):
        # print(f'Resetting data ...')
        self.data = self.data_temp.copy()
        if not self.trim_data(X, threshold=threshold):
            return False
        self.get_ctypes()
        return True
        # print(f'Data reset!')

    def init_properties(self):
        # print('Initializing properties ...')
        u = self.data['x_velocity'].values
        v = self.data['y_velocity'].values
        w = self.data['z_velocity'].values
        gx = self.data['progress_variable_gx'].values
        gy = self.data['progress_variable_gy'].values
        gz = self.data['progress_variable_gz'].values
        self.flat_X = self.X.flatten()
        self.U = np.array([u, v, w]).T
        self.flat_U = self.U.flatten()
        self.grad = np.array([gx, gy, gz]).T
        self.flat_grad = self.grad.flatten()

    def triangulation(self):
        # print('Delaunay Triangulation ...')
        x = self.data['X'].values
        y = self.data['Y'].values
        z = self.data['Z'].values
        self.X = np.array([x, y]).T
        self.tri = Delaunay(self.X)
        self.X = np.array([x, y, z]).T
        self.simplex_vertices = self.X[self.tri.simplices]
        self.flat_simplices = self.tri.simplices.flatten()

    def get_ctypes(self):
        # print('Converting to C arrays ...')
        self.c_simplicies = self.flat_simplices.ctypes.data_as(IntPtr)
        self.c_X = self.flat_X.ctypes.data_as(DoublePtr)
        self.c_U = self.flat_U.ctypes.data_as(DoublePtr)
        self.c_K = self.K.ctypes.data_as(DoublePtr)
        self.c_T = self.T.ctypes.data_as(DoublePtr)
        self.c_r = self.r.ctypes.data_as(DoublePtr)
        self.c_grad = self.flat_grad.ctypes.data_as(DoublePtr)


def deltaPoint(isoT0, isoT1, index=None):
    dt = isoT1.t - isoT0.t
    point_index = index
    if index is None:
        point_index = random.randint(0, isoT0.num_pts - 1)
    point_X = np.copy(isoT0.X[point_index]).flatten()
    c_point_X = point_X.ctypes.data_as(DoublePtr)
    point_U = np.copy(isoT0.U[point_index]).flatten()
    c_point_U = point_U.ctypes.data_as(DoublePtr)
    point_grad = np.copy(isoT0.grad[point_index]).flatten()
    c_point_grad = point_grad.ctypes.data_as(DoublePtr)
    point_K = isoT0.K[point_index]
    point_r = 1 / point_K
    point_T = isoT0.T[point_index]
    status = isoT1.reset_point(point_X)
    if not status:
        deltaK = -100000
        delta_r = -100000
        drdt = -100000
        kdrdt = -100000
    else:
        deltaK = getDelta(c_point_X, c_point_U, c_point_grad, point_K, isoT1.c_simplicies, isoT1.c_X, isoT1.c_U,
                          isoT1.c_K,
                          isoT1.num_simplices, isoT1.num_pts, dt)
        delta_r = (1 / (point_K + deltaK)) - (1 / point_K)

        drdt = delta_r / dt
        kdrdt = point_K * drdt

        if np.isnan(delta_r) or np.isinf(delta_r): deltaK = -100000
    # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
    return {'dK': deltaK,
            'dr': delta_r,
            'K': point_K,
            'r': point_r,
            'temp': point_T,
            'drdt': drdt,
            'kdrdt': kdrdt,
            'index': point_index,
            'dt': dt
            }


def delta_r_Point(isoT0, isoT1, index=None):
    dt = isoT1.t - isoT0.t
    point_index = index
    if index is None:
        point_index = random.randint(0, isoT0.num_pts - 1)
    point_X = np.copy(isoT0.X[point_index]).flatten()
    c_point_X = point_X.ctypes.data_as(DoublePtr)
    point_U = np.copy(isoT0.U[point_index]).flatten()
    c_point_U = point_U.ctypes.data_as(DoublePtr)
    point_grad = np.copy(isoT0.grad[point_index]).flatten()
    c_point_grad = point_grad.ctypes.data_as(DoublePtr)
    point_K = isoT0.K[point_index]
    point_r = isoT0.r[point_index]
    point_T = isoT0.T[point_index]
    status = isoT1.reset_point(point_X)
    if not status:
        delta_r = -100000
        drdt = -100000
        kdrdt = -100000
    else:
        delta_r = getDelta(c_point_X, c_point_U, c_point_grad, point_r, isoT1.c_simplicies, isoT1.c_X, isoT1.c_U, isoT1.c_r,
                          isoT1.num_simplices, isoT1.num_pts, dt)

        drdt = delta_r/dt
        kdrdt = point_K * drdt
    # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
    return {'dr': delta_r,
            'K': point_K,
            'r': point_r,
            'temp': point_T,
            'drdt': drdt,
            'kdrdt': kdrdt,
            'index': point_index,
            'dt': dt
            }


def delta_r_Point_test(isoT0, isoT1, index=None):
    dt = isoT1.t - isoT0.t
    point_index = index
    if index is None:
        point_index = random.randint(0, isoT0.num_pts - 1)
    point_X = np.copy(isoT0.X[point_index]).flatten()
    c_point_X = point_X.ctypes.data_as(DoublePtr)
    point_U = np.copy(isoT0.U[point_index]).flatten()
    c_point_U = point_U.ctypes.data_as(DoublePtr)
    point_grad = np.copy(isoT0.grad[point_index]).flatten()
    c_point_grad = point_grad.ctypes.data_as(DoublePtr)
    point_K = isoT0.K[point_index]
    point_r = isoT0.r[point_index]
    point_T = isoT0.T[point_index]
    status = isoT1.reset_point(point_X)
    if not status:
        delta_r = -100000
        drdt = -100000
        kdrdt = -100000
        X = [0, 0, 0]
    else:
        res = getDeltaPoint(c_point_X, c_point_U, c_point_grad, point_r, isoT1.c_simplicies, isoT1.c_X, isoT1.c_U, isoT1.c_r,
                          isoT1.num_simplices, isoT1.num_pts, dt)
        delta_res = np.ctypeslib.as_array(res, shape=(4,))
        if delta_res[0] == -100000:
            delta_r = -100000
            drdt = -100000
            kdrdt = -100000
            X = [0, 0, 0]
        else:
            delta_r = delta_res[3]
            X = [delta_res[0], delta_res[1], delta_res[2]]
            drdt = delta_r/dt
            kdrdt = point_K * drdt
    # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
    return {'dr': delta_r,
            'K': point_K,
            'r': point_r,
            'temp': point_T,
            'drdt': drdt,
            'kdrdt': kdrdt,
            'index': point_index,
            'dt': dt,
            'x0': point_X[0],
            'y0': point_X[1],
            'z0': point_X[2],
            'x1': X[0],
            'y1': X[1],
            'z1': X[2]
            }


def main():
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    print('Finished constructing isosurfaces!')
    delta_list = []
    # num_rands = 100000
    # for i in range(num_rands):
    for i in range(len(isoT0.data)):
        if i % 1000 == 0:
            print(f'getting delta for random particle {i}')
        delta = delta_r_Point(isoT0, isoT1, i)
        # delta = deltaIndexPoint(isoT0, isoT1, i)
        if delta['dr'] != -100000:  # and delta not in delta_list:
            delta_list.append(delta)

    out_df = pd.DataFrame.from_dict(delta_list)
    print(out_df)
    out_df.to_pickle('res/TemporalPathData_fourth_last.pkl')
    print('Done!')
    plotDataPDF(out_df, 'K', 'kdrdt')
    return


def plotDataPDF(data, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    pdf_2D(fig, ax, data[x_label], data[y_label], x_label, y_label, dpi=75)
    plt.show()


def plotDataHeatMap(data, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_against_2D(fig, ax, data[x_label], data[y_label], data[z_label], x_label, y_label, z_label)
    plt.show()


def plotPointTrajectories(data):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('start_data')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('r_data')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('dr_data')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('surface_data')
    # ax6 = fig.add_subplot(2, 3, 6, projection='scatter_density')
    # ax6.set_title('kdrdt_data')
    bins = [150, 150]
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT0 = IsoSurface('res/testIso01920.dat')
    # plot_against_2D(fig, ax1,
    #                 data['x0'], data['y0'], data['z0'],
    #                 'x', 'y', 'z', bins=bins, scale='log')
    plot_against_2D(fig, ax1,
                    data['x1'], data['y1'], data['z1'],
                    'x', 'y', 'z', bins=bins, scale='log')
    plot_against_2D(fig, ax2,
                    data['x0'], data['y0'], data['r'],
                    'x', 'y', 'r', bins=bins, scale='log')
    plot_against_2D(fig, ax3,
                    data['x1'], data['y1'], abs(data['dr']),
                    'x', 'y', 'dr', bins=bins, scale='log')
    # plot_against_2D(fig, ax4,
                    # data['x1'], data['y1'], abs(data['K']),
                    # 'x', 'y', 'k', bins=bins, scale='log')
    plot_against_2D(fig, ax4,
                    isoT0.X.T[0], isoT0.X.T[1], (isoT0.data['progress_variable_gx']**2 + isoT0.data['progress_variable_gy']**2 + isoT0.data['progress_variable_gz']**2)**0.5,
                    'x', 'y', '|g|', bins=bins, scale='log')
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


def getPointPath():
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.016, 0.032], ylim=[0.016, 0.032])
    # isoT0 = IsoSurface('res/testIso01920.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT1 = IsoSurface('res/testIso01930.dat', xlim=[0.0, 0.016], ylim=[0.0, 0.016])
    # isoT0 = IsoSurface('res/testIso01920.dat')
    # isoT1 = IsoSurface('res/testIso01930.dat')
    print('Finished constructing isosurfaces!')
    delta_list = []
    # num_rands = 100000
    # for i in range(num_rands):
    num_invalid = 0
    for i in range(len(isoT0.data)):
        if i % 1000 == 0:
            print(f'getting delta for random particle {i}')
            print(f'num invalid: {num_invalid}')
        delta = delta_r_Point_test(isoT0, isoT1, i)
        # delta = deltaIndexPoint(isoT0, isoT1, i)
        if delta['dr'] != -100000:  # and delta not in delta_list:
            delta_list.append(delta)
        else:
            num_invalid += 1

    out_df = pd.DataFrame.from_dict(delta_list)
    print(out_df)
    out_df.to_pickle('res/TemporalPathData_all.pkl')
    print('Done!')
    plotDataPDF(out_df, 'K', 'kdrdt')
    return


def readPkl():
    file_name = 'res/TemporalPathData_all.pkl'
    data = pd.read_pickle(file_name)
    print(data)
    print(f'read {len(data)} rows of data')
    print(f'r:\n{data["r"]}')
    print(f'max r: {max(data["r"])}')
    print(f'min r: {min(data["r"])}')
    print(f'dr:\n{data["dr"]}')
    print(f'max dr: {max(data["dr"])}')
    print(f'min dr: {min(data["dr"])}')
    plotPointTrajectories(data)
    # plotDataPDF(data, 'r', 'dr')
    # plotDataPDF(data, 'K', 'kdrdt')
    # plotDataHeatMap(data, 'K', 'kdrdt', 'temp')


def particlePath(point_index=None):
    iso_range = [f'res/path_out/pathIso0{i}' for i in range(1000, 2040, 10)]
    path_list = []
    firstIter = True
    for i in range(len(iso_range) - 1):
        isoT0 = IsoSurface(iso_range[i])
        isoT1 = IsoSurface(iso_range[i+1])
        dt = isoT1.t - isoT0.t
        if point_index is None and firstIter:
            point_index = random.randint(0, isoT0.num_pts - 1)

        point_X = np.copy(isoT0.X[point_index]).flatten()
        c_point_X = point_X.ctypes.data_as(DoublePtr)
        point_U = np.copy(isoT0.U[point_index]).flatten()
        c_point_U = point_U.ctypes.data_as(DoublePtr)
        point_grad = np.copy(isoT0.grad[point_index]).flatten()
        c_point_grad = point_grad.ctypes.data_as(DoublePtr)
        point_K = isoT0.K[point_index]
        point_r = 1 / point_K
        point_T = isoT0.T[point_index]
        status = isoT1.reset_point(point_X)

        if not status:
            deltaK = -100000
            delta_r = -100000
            drdt = -100000
            kdrdt = -100000
        else:
            deltaK = getDelta(c_point_X, c_point_U, c_point_grad, point_K, isoT1.c_simplicies, isoT1.c_X, isoT1.c_U,
                              isoT1.c_K,
                              isoT1.num_simplices, isoT1.num_pts, dt)
            delta_r = (1 / (point_K + deltaK)) - (1 / point_K)

            drdt = delta_r / dt
            kdrdt = point_K * drdt
            if np.isnan(delta_r) or np.isinf(delta_r): deltaK = -100000
        # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
        path_list.append({'dK': deltaK,
                          'dr': delta_r,
                          'K': point_K,
                          'r': point_r,
                          'temp': point_T,
                          'drdt': drdt,
                          'kdrdt': kdrdt,
                          'index': point_index,
                          'dt': dt})

        firstIter = False


if __name__ == '__main__':
    # main()
    # getPointPath()
    readPkl()
    # formatCSV('res/01920Lev3.csv')
    # formatCSV('res/01930Lev3.csv')
    # CSVtoPKL('01920Lev3')
    # CSVtoPKL('01930Lev3')
    # df = df[(df['profgress_variable'] < 0.825) & (df['progress_variable'] > 0.775)]
    # pd.to_pickle(df, 'res/iso01920Lev3.pkl')
    # df1 = pd.read_pickle('res/01930Lev3.pkl')
    # df1['r'] = abs(df1['r'])
    # pd.to_pickle(df1, 'res/01930Lev3.pkl')

    # time, df2 = ReadDat('res/testIso01920.dat')
    # print(df1.columns)
    # print(df2.columns)
