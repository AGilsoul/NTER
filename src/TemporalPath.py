from ctypes import cdll, c_int, c_float, POINTER, c_bool, c_void_p
from formatDAT import formatDAT
from ReadDat import ReadDat
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

FloatPtr = POINTER(c_float)
IntPtr = POINTER(c_int)

# lib_path = 'lib/GradAscent.so'  # linux
lib_path = 'lib/FlameIsoPath.dll'  # windows
GradAscentLib = cdll.LoadLibrary(lib_path)
testRun = GradAscentLib.testRun

testRun.argtypes = [IntPtr,
                    IntPtr,
                    FloatPtr,
                    FloatPtr,
                    FloatPtr,
                    FloatPtr,
                    FloatPtr,
                    IntPtr,
                    IntPtr]
testRun.restype = c_int


class IsoSurface:
    def __init__(self, file_name, xlim=0.032, ylim=0.032):
        print(f'Constructing IsoSurface from {file_name} ...')
        formatDAT(file_name)
        self.t, self.data = ReadDat(file_name)
        self.data = self.data[(self.data['X'] < xlim) & (self.data['Y'] < ylim)]
        x = self.data['X'].values
        y = self.data['Y'].values
        z = self.data['Z'].values
        K = np.array(self.data['MeanCurvature_progress_variable'])
        T = np.array(self.data['temp'])

        points = np.array([x, y]).T
        tri = Delaunay(points)
        new_pts = np.array([x, y, z]).T
        simplex_vertices = new_pts[tri.simplices]
        num_pts = len(new_pts)
        num_simplices = len(simplex_vertices)
        u = self.data['x_velocity'].values
        v = self.data['y_velocity'].values
        w = self.data['z_velocity'].values
        gx = self.data['progress_variable_gx'].values
        gy = self.data['progress_variable_gy'].values
        gz = self.data['progress_variable_gz'].values
        self.flat_simplices = tri.simplices.flatten()
        X = new_pts.flatten()
        U = np.array([u, v, w]).T.flatten()
        grad = np.array([gx, gy, gz]).T.flatten()
        print(f'\t{num_simplices} simplices')
        print(f'\t{num_pts} points')
        self.c_simplicies = self.flat_simplices.ctypes.data_as(IntPtr)
        self.c_X = X.ctypes.data_as(FloatPtr)
        self.c_U = U.ctypes.data_as(FloatPtr)
        self.c_K = K.ctypes.data_as(FloatPtr)
        self.c_T = T.ctypes.data_as(FloatPtr)
        self.c_grad = grad.ctypes.data_as(FloatPtr)
        self.ptsPer = num_pts
        self.simpsPer = num_simplices
        self.simplex_vertices = simplex_vertices


def main():
    isoT0 = IsoSurface('res/testIso01920.dat', xlim=0.008, ylim=0.008)
    isoT1 = IsoSurface('res/testIso01930.dat', xlim=0.008, ylim=0.008)
    c_simpsPerSurf = np.array([isoT0.simpsPer, isoT1.simpsPer]).ctypes.data_as(IntPtr)
    c_ptsPerSurf = np.array([isoT0.ptsPer, isoT1.ptsPer]).ctypes.data_as(IntPtr)

    print(f'isoT0 simplices:\n{isoT0.simplex_vertices}')
    print(f'isoT1 simplices:\n{isoT1.simplex_vertices}')
    print(f'isoT0 simplex indices:\n{isoT0.flat_simplices}\n')
    testRun(isoT0.c_simplicies, isoT1.c_simplicies, isoT0.c_X, isoT1.c_X, isoT0.c_U, isoT1.c_U, isoT0.c_grad, c_simpsPerSurf, c_ptsPerSurf)
    print('Done!')
    return


if __name__ == '__main__':
    main()

