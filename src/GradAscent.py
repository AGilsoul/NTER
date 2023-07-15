from ctypes import cdll, c_int, c_float, POINTER
import modin.pandas as pd
import numpy as np
from multiprocessing import freeze_support
from modin import config
from distributed import Client
from csvToPkl import CSVtoPKL

FloatPtr = POINTER(c_float)
IntPtr = POINTER(c_int)

lib_path = 'lib/GradAscent.dll'
GradAscentLib = cdll.LoadLibrary(lib_path)
gradAscent = GradAscentLib.getNextPoints
gradAscent.argtypes = [FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               c_int]
gradAscent.restype = FloatPtr
fillGrid = GradAscentLib.fillGrid
fillGrid.argtypes = [FloatPtr,
                             FloatPtr,
                             FloatPtr,
                             FloatPtr,
                             FloatPtr,
                             FloatPtr,
                             FloatPtr,
                             IntPtr,
                             c_int]
fillGrid.restype = FloatPtr


class GradAscent:
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
        gradAscent.argtypes = [FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               c_int]
        gradAscent.restype = FloatPtr
        res_pts = gradAscent(c_pts, c_gx, c_gy, c_gz, self.num_pts)
        res = np.ctypeslib.as_array(res_pts, shape=(self.num_pts*3,))
        return res


class AMRGrid:
    def __init__(self, data):
        self.amr_data = data
        self.cell_pos = [self.amr_data['CellCenters:0'],
                         self.amr_data['CellCenters:1'],
                         self.amr_data['CellCenters:2']]
        self.cell_grads = [self.amr_data['progress_variable_gx'],
                           self.amr_data['progress_variable_gy'],
                           self.amr_data['progress_variable_gz']]
        self.cell_prog = self.amr_data['progress_variable']
        self.pos_sorted_unique = None
        self.pos_dif = None
        self.pos_lims = None
        self.num_cells = None
        self.amr_shape = None
        self.c_x = None
        self.c_y = None
        self.c_z = None
        self.c_xyz_mins = None
        self.c_xyz_difs = None
        self.c_grid_dims = None
        self.GA = None
        self.setFilledAMRShape()
        self.amr_prog_grid = self.fillAMRGrid(self.amr_prog_grid, self.cell_prog, 'progress_variable')
        # self.fillAMRGrid()
        # print('done!')

    @staticmethod
    def getSortedUniqueDataDif(data):
        return data[1] - data[0]

    @staticmethod
    def getSortedUniqueDataLims(data):
        return data[0], data[-1]

    def cellPosToGridIndex(self, pos):
        return [round((pos[i] - self.pos_lims[i][0]) / self.pos_dif[i]) for i in range(len(pos))]

    def gridIndexToCellPos(self, index):
        return [index[i] * self.pos_dif[i] + self.pos_lims[i][0] for i in range(len(index))]

    # gets shape for full AMR grid
    def setFilledAMRShape(self):
        self.pos_sorted_unique = [sorted(x.unique()) for x in self.cell_pos]
        self.pos_dif = np.array([self.getSortedUniqueDataDif(x) for x in self.pos_sorted_unique], dtype=np.float32)
        self.pos_lims = np.array([self.getSortedUniqueDataLims(x) for x in self.pos_sorted_unique], dtype=np.float32)
        self.amr_shape = np.array([round((self.pos_lims[i][1] - self.pos_lims[i][0]) / self.pos_dif[i]) for i in range(3)], dtype=np.int32)
        self.amr_prog_grid = np.zeros(self.amr_shape)

        self.c_x = self.listToFloatPtr(self.cell_pos[0])
        self.c_y = self.listToFloatPtr(self.cell_pos[1])
        self.c_z = self.listToFloatPtr(self.cell_pos[2])
        xyz_mins = np.array([i[0] for i in self.pos_lims], dtype=np.float32)
        self.c_xyz_mins = self.listToFloatPtr(xyz_mins)
        self.c_xyz_difs = self.listToFloatPtr(self.pos_dif)
        self.c_grid_dims = self.listToIntPtr(self.amr_shape)

    @staticmethod
    def listToFloatPtr(d):
        return np.array(d, dtype=np.float32).ctypes.data_as(FloatPtr)

    @staticmethod
    def listToIntPtr(d):
        return np.array(d, dtype=np.int32).ctypes.data_as(IntPtr)

    def gradAscentPoint(self):
        p_i = 1000
        p_i_xyz = [self.cell_pos[0][p_i], self.cell_pos[1][p_i], self.cell_pos[2][p_i]]
        print(f'original coords: {p_i_xyz}')
        gx = [self.cell_grads[0][p_i]]
        gy = [self.cell_grads[1][p_i]]
        gz = [self.cell_grads[2][p_i]]
        print(f'cur gradient: ({gx},{gy},{gz})')
        num_pts = 1
        self.GA = GradAscent(p_i_xyz, gx, gy, gz, num_pts)
        res_pts = self.GA.perform_gA()
        print(res_pts)

    def fillAMRGrid(self, grid_to_fill, val_to_fill, val_name):
        print(f'filling {val_name} grid ...')
        flat_grid = grid_to_fill.flatten()
        grid_size = len(flat_grid)
        c_grid = self.listToFloatPtr(flat_grid)
        c_val = self.listToFloatPtr(val_to_fill)

        res_pts = fillGrid(c_grid,
                           c_val,
                           self.c_x,
                           self.c_y,
                           self.c_z,
                           self.c_xyz_mins,
                           self.c_xyz_difs,
                           self.c_grid_dims,
                           len(self.amr_data))

        grid_to_fill = np.ctypeslib.as_array(res_pts, shape=(grid_size,)).reshape(grid_to_fill.shape)
        print('Done!')
        return grid_to_fill


if __name__ == '__main__':
    # CSVtoPKL('combGradCurv01929Data')
    file_name = 'res/combGradCurv01929Data.pkl'
    client = Client()
    freeze_support()
    config.MinPartitionSize.put(128)
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    print(f'Setting up grid ...')
    grid = AMRGrid(amr_data)
    # grid.gradAscentPoint()
