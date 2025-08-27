import ctypes
import random
from ctypes import cdll, c_int, c_double, POINTER, c_bool, c_void_p
from formatDAT import formatDAT
from ReadDat import ReadDat
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import mpl_scatter_density
from ProbDistFunc import pdf_2D, plot_against_2D
import statistics
import plotly.graph_objects as go
from mayavi import mlab

DoublePtr = POINTER(c_double)
IntPtr = POINTER(c_int)

# lib_path = 'lib/GradAscent.so'  # linux
lib_path = 'lib/FlameIsoPath.dll'  # windows
GradAscentLib = cdll.LoadLibrary(lib_path)
getDeltas = GradAscentLib.getDeltas
getAvgTriVals = GradAscentLib.getAvgTriVals

getDeltas.argtypes = [IntPtr,
                      IntPtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      DoublePtr,
                      IntPtr,
                      IntPtr,
                      c_double]
getDeltas.restype = DoublePtr

getAvgTriVals.argtypes = [
    IntPtr,
    DoublePtr,
    DoublePtr,
    DoublePtr,
    c_int,
    c_int
]
getAvgTriVals.restype = DoublePtr


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
        hull = Delaunay(self.X)
        self.X = np.array([x, y, z]).T
        # hull = ConvexHull(self.X)
        self.simplex_vertices = self.X[hull.simplices]
        print(self.simplex_vertices)
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.set_xlim([0,xlim])
        # ax.set_ylim([0,ylim])
        # ax.set_zlim([min(z), max(z)])
        # pc = a3.art3d.Poly3DCollection(self.simplex_vertices, edgecolor='black')
        # ax.add_collection(pc)
        # ax.plot_trisurf(self.X.T[0], self.X.T[1], self.X.T[2], triangles=hull.simplices)
        # ax.plot_trisurf(self.X.T[0], self.X.T[1], self.X.T[2])
        # ax.plot(self.X[0], self.X[1], self.X[2])
        plt.show()
        self.num_pts = len(self.X)
        self.num_simplices = len(self.simplex_vertices)
        u = self.data['x_velocity'].values
        v = self.data['y_velocity'].values
        w = self.data['z_velocity'].values
        gx = self.data['progress_variable_gx'].values
        gy = self.data['progress_variable_gy'].values
        gz = self.data['progress_variable_gz'].values
        self.flat_simplices = hull.simplices.flatten()
        self.flat_X = self.X.flatten()
        self.U = np.array([u, v, w]).T
        self.flat_U = self.U.flatten()
        self.grad = np.array([gx, gy, gz]).T
        self.flat_grad = self.grad.flatten()
        self.c_simplicies, self.c_X, self.c_U, self.c_K, self.c_T, self.c_grad = None, None, None, None, None, None
        self.get_ctypes()

    def get_ctypes(self):
        self.c_simplicies = np.copy(self.flat_simplices).ctypes.data_as(IntPtr)
        self.c_X = np.copy(self.flat_X).ctypes.data_as(DoublePtr)
        self.c_U = np.copy(self.flat_U).ctypes.data_as(DoublePtr)
        self.c_K = np.copy(self.K).ctypes.data_as(DoublePtr)
        self.c_T = np.copy(self.T).ctypes.data_as(DoublePtr)
        self.c_grad = np.copy(self.flat_grad).ctypes.data_as(DoublePtr)


def main():
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=0.01, ylim=0.01)
    print(np.unique(isoT0.X.T[0]))
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=0.01, ylim=0.01)
    dt = isoT1.t - isoT0.t

    num_tri = np.array([isoT0.num_simplices, isoT1.num_simplices])
    num_pts = np.array([isoT0.num_pts, isoT1.num_pts])
    c_tri = num_tri.ctypes.data_as(IntPtr)
    c_pts = num_pts.ctypes.data_as(IntPtr)

    print("Getting deltas ...")
    deltaK = getDeltas(isoT0.c_simplicies, isoT1.c_simplicies, isoT0.c_X, isoT1.c_X, isoT0.c_U, isoT1.c_U, isoT0.c_K, isoT1.c_K, isoT0.c_grad, isoT1.c_grad,
                       c_tri, c_pts, dt)
    deltaK = np.ctypeslib.as_array(deltaK, shape=(isoT0.num_simplices,))
    del isoT1
    isoT0.get_ctypes()
    avgT = getAvgTriVals(isoT0.c_simplicies, isoT0.c_X, isoT0.c_U, isoT0.c_T, isoT0.num_simplices, isoT0.num_pts)
    avgT = np.ctypeslib.as_array(avgT, shape=(isoT0.num_simplices,))

    k_indices = np.argwhere(deltaK == -100000)
    deltaK = np.delete(deltaK, k_indices)
    avgT = np.delete(avgT, k_indices)
    T_indices = np.argwhere(avgT == -100000)
    deltaK = np.delete(deltaK, T_indices)
    avgT = np.delete(avgT, T_indices)
    '''
    temp_indices = np.argwhere(avgT < 520)
    deltaK = np.delete(deltaK, temp_indices)
    avgT = np.delete(avgT, temp_indices)
    '''

    # deltaR = np.array([(1/(avgK[i] + deltaK[i])) - (1/avgK[i]) if avgK[i] != 0 else 0 for i in range(len(deltaK))])

    print(deltaK)
    print(avgT)
    print(f'triangle index 58207 vertices: ({isoT0.X[25011]}, {isoT0.X[25008]}, {isoT0.X[25007]})')
    print(f'triangle index 58207 vals: ({isoT0.K[25011]}, {isoT0.K[25008]}, {isoT0.K[25007]})')
    print('Done!')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # ax2 = fig.add_subplot(2, 1, 2, projection='scatter_density')
    pdf_2D(fig, ax1, avgT, deltaK, 'temperature T (k)', 'dK (1/m)', dpi=50)
    # pdf_2D(fig, ax2, avgT, deltaR/dt, 'temperature T (K)', 'dR/dt (m/s)', dpi=35)

    # plot_against_2D(fig, ax, avgT, avgK, deltaK/dt, 'temperature T (K)', 'Curvature K (1/m)', 'dK/dt')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    main()