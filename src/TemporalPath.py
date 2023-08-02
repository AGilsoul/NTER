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
testRun.restype = c_void_p


def main():
    file_name = 'res/testIso01920.dat'
    # formatDAT(file_name)
    t, data = ReadDat(file_name)
    print(f'Surface at t={t}:')
    print(data)
    X = data['X'].values
    Y = data['Y'].values
    Z = data['Z'].values

    points = np.array([X, Y]).T
    print(f'points:\n{points}')
    tri = Delaunay(points)
    new_pts = np.array([X, Y, Z]).T
    simplex_vertices = new_pts[tri.simplices]
    print(f'simplices:\n{simplex_vertices}')
    num_pts = len(new_pts)
    num_simplices = len(simplex_vertices)
    print(f'num points: {num_pts}')
    print(f'num simplices: {num_simplices}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap=plt.cm.Spectral)
    plt.show()

    return


if __name__ == '__main__':
    main()

