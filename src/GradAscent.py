from ctypes import cdll, c_int, c_float, POINTER
import modin.pandas as pd
import numpy as np
from multiprocessing import freeze_support
from modin import config
from distributed import Client
from csvToPkl import CSVtoPKL

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


class AMRGrid:
    def __init__(self, data):
        self.amr_data = data
        self.cell_pos = None
        self.pos_sorted_unique = None
        self.pos_dif = None
        self.pos_lims = None
        self.num_cells = None
        self.amr_shape = None
        self.cell_grads = None
        self.amr_pos_grid = None
        self.amr_gradient_grid = None

    @staticmethod
    def getSortedUniqueDataDif(data):
        return data[1] - data[0]

    @staticmethod
    def getSortedUniqueDataLims(data):
        return data[0], data[-1]

    def cellPosToGridIndex(self, pos):
        return [(pos[i] - self.pos_lims[i][0]) / self.pos_dif[i] for i in range(len(pos))]

    def gridIndexToCellPos(self, index):
        return [index[i] * self.pos_dif[i] + self.pos_lims[i][0] for i in range(len(index))]

    # gets shape for full AMR grid
    def setFilledAMRShape(self):
        self.cell_pos = [self.amr_data["CellCenters:0"],
                         self.amr_data["CellCenters:1"],
                         self.amr_data["CellCenters:2"]]
        self.pos_sorted_unique = [sorted(x.unique()) for x in self.cell_pos]
        self.pos_dif = [self.getSortedUniqueDataDif(x) for x in self.pos_sorted_unique]
        self.pos_lims = [self.getSortedUniqueDataLims(x) for x in self.pos_sorted_unique]
        self.num_cells = [round((self.pos_lims[i][1] - self.pos_lims[i][0]) / self.pos_dif[i]) for i in range(3)]
        self.amr_shape = self.num_cells[0], self.num_cells[1], self.num_cells[2]

    def fillAMRGrid(self):
        self.cell_grads = [self.amr_data["progress_variable_gx"],
                           self.amr_data["progress_variable_gy"],
                           self.amr_data["progress_variable_gz"]]
        self.setFilledAMRShape()
        shape = self.amr_shape
        print(f'amr shape: {shape}')
        self.amr_pos_grid = np.zeros(shape=shape)
        self.amr_gradient_grid = np.zeros(shape=(3,shape[0], shape[1], shape[2]))
        print(self.amr_pos_grid)
        print(f'grad shape: {self.amr_gradient_grid.shape}')


if __name__ == '__main__':
    # CSVtoPKL('combGradCurv01929Data')
    file_name = 'res/combGradCurv01929Data.pkl'
    client = Client()
    freeze_support()
    config.MinPartitionSize.put(128)
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    grid = AMRGrid(amr_data)
    print('Filling grid...')
    grid.fillAMRGrid()
