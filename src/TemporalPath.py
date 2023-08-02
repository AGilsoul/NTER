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
                    FloatPtr,
                    FloatPtr,
                    FloatPtr,
                    FloatPtr,
                    IntPtr,
                    IntPtr,
                    c_int]
testRun.restype = c_int


def main():
    file_name = 'res/testIso01920.dat'
    # formatDAT(file_name)
    t, data = ReadDat(file_name)
    print(f'Surface at t={t}:')
    print(data)
    x = data['X'].values
    y = data['Y'].values
    z = data['Z'].values

    points = np.array([x, y]).T
    print(f'points:\n{points}')
    tri = Delaunay(points)
    new_pts = np.array([x, y, z]).T
    simplex_vertices = new_pts[tri.simplices]
    print(f'simplex indices:\n{tri.simplices}')
    print(f'simplices:\n{simplex_vertices}')
    num_pts = len(new_pts)
    num_simplices = len(simplex_vertices)
    print(f'num points: {num_pts}')
    print(f'num simplices: {num_simplices}')

    u = data['x_velocity'].values
    v = data['y_velocity'].values
    w = data['z_velocity'].values
    gx = data['progress_variable_gx'].values
    gy = data['progress_variable_gy'].values
    gz = data['progress_variable_gz'].values
    contour_times = np.array([t])
    simpPer = np.array([num_simplices])
    partPer = np.array([num_pts])
    num_surfaces = 1

    flat_simplices = tri.simplices.flatten()
    X = new_pts.flatten()
    U = np.array([u, v, w]).T.flatten()
    grad = np.array([gx, gy, gz]).T.flatten()

    c_simplicies = flat_simplices.ctypes.data_as(IntPtr)
    c_X = X.ctypes.data_as(FloatPtr)
    c_U = U.ctypes.data_as(FloatPtr)
    c_grad = grad.ctypes.data_as(FloatPtr)
    c_times = contour_times.ctypes.data_as(FloatPtr)
    c_simp_per = simpPer.ctypes.data_as(IntPtr)
    c_part_per = partPer.ctypes.data_as(IntPtr)

    print('\ntest run ...\n')
    testRun(c_simplicies, c_X, c_U, c_grad, c_times, c_simp_per, c_part_per, num_surfaces)
    print('Done!')
    return


if __name__ == '__main__':
    main()

