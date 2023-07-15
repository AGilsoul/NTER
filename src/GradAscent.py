from ctypes import cdll, c_int, c_float, POINTER
import modin.pandas as pd
import numpy as np
from multiprocessing import freeze_support
from modin import config
from distributed import Client

lib_path = 'lib/GradAscent.dll'
GradAscentLib = cdll.LoadLibrary(lib_path)
GradAscent = GradAscentLib.getNextPoints

FloatPtr = POINTER(c_float)


class gradAscent:
    def __init__(self, pts, gx, gy, gz, num_pts):
        self.pts = np.array(pts, dtype=np.float32)
        self.gx = np.array(gx, dtype=np.float32)
        self.gy = np.array(gy, dtype=np.float32)
        self.gz = np.array(gz, dtype=np.float32)
        self.num_pts = num_pts

    def perform_gA(self):
        c_pts = self.pts.ctypes.data_as(FloatPtr)
        c_gx = self.gx.ctypes.data_as(FloatPtr)
        c_gy = self.gy.ctypes.data_as(FloatPtr)
        c_gz = self.gz.ctypes.data_as(FloatPtr)

        GradAscent.argtypes = [FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               c_int]
        GradAscent.restype = FloatPtr

        res_pts = GradAscent(c_pts, c_gx, c_gy, c_gz, self.num_pts)
        res = np.ctypeslib.as_array(res_pts, shape=(self.num_pts*3,))


def fillAMRGrid(amr_data):
    px, py, pz = amr_data["CellCenters:0"], \
                 amr_data["CellCenters:1"], \
                 amr_data["CellCenters:2"]

    gx, gy, gz = amr_data["progress_variable_gx"], \
                 amr_data["progress_variable_gx"], \
                 amr_data["progress_variable_gx"]

    all_x, all_y, all_z = sorted(px.unique()), \
                          sorted(py.unique()), \
                          sorted(pz.unique())
    print('unique vals: ')
    print(all_x)
    print(all_y)
    print(all_z)

    prog_var = amr_data["progress_variable"]

    x_lim = (min(px), max(px))
    y_lim = (min(py), max(py))
    z_lim = (min(pz), max(pz))

    print(f'prog min,max : {min(prog_var)},{max(prog_var)}')
    lims = [x_lim, y_lim, z_lim]
    print(f'xyz lims: {lims}')
    num_x = lims[0][1] / lims[0][0]
    num_y = lims[1][1] / lims[1][0]
    print(f'num x: {num_x}, num y: {num_y}')

    return


if __name__ == '__main__':
    file_name = 'res/combGradCurv01929Data.pkl'
    client = Client()
    freeze_support()
    config.MinPartitionSize.put(128)
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    fillAMRGrid(amr_data)
