import ctypes
import random
from ctypes import cdll, c_int, c_double, POINTER, c_bool, c_void_p
from formatDAT import formatDAT
from ReadDat import ReadDat
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mpl_scatter_density
from ProbDistFunc import pdf_2D, plot_against_2D
import statistics

DoublePtr = POINTER(c_double)
IntPtr = POINTER(c_int)

# lib_path = 'lib/GradAscent.so'  # linux
lib_path = 'lib/CPUSingleTemporalPath.dll'  # windows
FlamePathLib = cdll.LoadLibrary(lib_path)
getDelta = FlamePathLib.getDelta

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


class IsoSurface:
    def __init__(self, file_name, xlim=0.032, ylim=0.032):
        print(f'Constructing IsoSurface from {file_name} ...')
        formatDAT(file_name)
        self.t, self.data = ReadDat(file_name)
        self.data = self.data[(self.data['X'] < xlim) & (self.data['Y'] < ylim)]
        x = self.data['X'].values
        y = self.data['Y'].values
        z = self.data['Z'].values
        self.K = np.array(self.data['MeanCurvature_progress_variable'])
        self.T = np.array(self.data['temp'])
        self.X = np.array([x, y]).T
        tri = Delaunay(self.X)
        self.X = np.array([x, y, z]).T
        self.simplex_vertices = self.X[tri.simplices]
        self.num_pts = len(self.X)
        self.num_simplices = len(self.simplex_vertices)
        u = self.data['x_velocity'].values
        v = self.data['y_velocity'].values
        w = self.data['z_velocity'].values
        gx = self.data['progress_variable_gx'].values
        gy = self.data['progress_variable_gy'].values
        gz = self.data['progress_variable_gz'].values
        self.flat_simplices = tri.simplices.flatten()
        self.flat_X = self.X.flatten()
        self.U = np.array([u, v, w]).T
        self.flat_U = self.U.flatten()
        self.grad = np.array([gx, gy, gz]).T
        self.flat_grad = self.grad.flatten()
        self.c_simplicies, self.c_X, self.c_U, self.c_K, self.c_T, self.c_grad = None, None, None, None, None, None
        self.get_ctypes()

    def get_ctypes(self):
        self.c_simplicies = self.flat_simplices.ctypes.data_as(IntPtr)
        self.c_X = self.flat_X.ctypes.data_as(DoublePtr)
        self.c_U = self.flat_U.ctypes.data_as(DoublePtr)
        self.c_K = self.K.ctypes.data_as(DoublePtr)
        self.c_T = self.T.ctypes.data_as(DoublePtr)
        self.c_grad = self.flat_grad.ctypes.data_as(DoublePtr)


def deltaRandomPoint(isoT0, isoT1, dt):
    point_index = random.randint(0, isoT0.num_pts - 1)
    point_X = np.copy(isoT0.X[point_index]).flatten()
    c_point_X = point_X.ctypes.data_as(DoublePtr)
    point_U = np.copy(isoT0.U[point_index]).flatten()
    c_point_U = point_U.ctypes.data_as(DoublePtr)
    point_grad = np.copy(isoT0.grad[point_index]).flatten()
    c_point_grad = point_grad.ctypes.data_as(DoublePtr)
    point_K = isoT0.K[point_index]
    point_r = 1/point_K
    point_T = isoT0.T[point_index]
    deltaK = getDelta(c_point_X, c_point_U, c_point_grad, point_K, isoT1.c_simplicies, isoT1.c_X, isoT1.c_U, isoT1.c_K,
                      isoT1.num_simplices, isoT1.num_pts, dt)
    delta_r = (1/(point_K + deltaK)) - (1/point_K)

    drdt = delta_r/dt
    kdrdt = point_K * drdt

    if np.isnan(delta_r) or np.isinf(delta_r): delta_r = -100000
    # deltaK = np.ctypeslib.as_array((c_double * int(len(isoT0.simplex_vertices) / 3)).from_address(ctypes.addressof(deltaK_ctypes.contents)))
    return {'dK': deltaK,
            'dr': delta_r,
            'K': point_K,
            'r': point_r,
            'temp': point_T,
            'drdt': drdt,
            'kdrdt': kdrdt,
            'index': point_index}


def main():
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=0.004, ylim=0.004)
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=0.004, ylim=0.004)
    dt = isoT1.t - isoT0.t

    delta_list = []
    num_rands = 10000
    for i in range(num_rands):
        if i % 1000 == 0:
            print(f'getting delta for random particle {i}')
        delta = deltaRandomPoint(isoT0, isoT1, dt)
        if delta['dK'] != -100000:
            delta_list.append(delta)
        else:
            i -= 1

    delta_list = np.array(delta_list).T
    # avgT = [isoT0.T[i] for i in range(len(deltaK_list[2]))]

    print(delta_list)
    print('Done!')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # ax2 = fig.add_subplot(2, 1, 2, projection='scatter_density')
    pdf_2D(fig, ax1, delta_list['temp'], delta_list['dK'], 'Temperature T (k)', 'dK', dpi=50)
    # pdf_2D(fig, ax2, avgT, deltaK/dt, 'temperature T (K)', 'dK/dt', dpi=35)

    # plot_against_2D(fig, ax, avgT, avgK, deltaK/dt, 'temperature T (K)', 'Curvature K (1/m)', 'dK/dt')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    main()