import math
from ctypes import cdll, c_int, c_float, POINTER
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

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
    @staticmethod
    def perform_gA(pts, gx, gy, gz, num_pts):
        pts = np.array(pts, dtype=np.float32)
        gx = np.array(gx, dtype=np.float32)
        gy = np.array(gy, dtype=np.float32)
        gz = np.array(gz, dtype=np.float32)
        num_pts = num_pts

        c_pts = pts.ctypes.data_as(FloatPtr)
        c_gx = gx.ctypes.data_as(FloatPtr)
        c_gy = gy.ctypes.data_as(FloatPtr)
        c_gz = gz.ctypes.data_as(FloatPtr)
        gradAscent.argtypes = [FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               c_int]
        gradAscent.restype = FloatPtr
        res_pts = gradAscent(c_pts, c_gx, c_gy, c_gz, num_pts)
        res = np.ctypeslib.as_array(res_pts, shape=(num_pts*3,))
        return res


class AMRGrid:
    def __init__(self, data):
        self.amr_data = data
        print(f'num points: {len(data)}')
        self.cell_pos = [self.amr_data['CellCenters:0'],
                         self.amr_data['CellCenters:1'],
                         self.amr_data['CellCenters:2']]
        self.cell_grads = [self.amr_data['progress_variable_gx'],
                           self.amr_data['progress_variable_gy'],
                           self.amr_data['progress_variable_gz']]
        self.cell_prog = self.amr_data['progress_variable']
        self.cell_curv = self.amr_data['MeanCurvature_progress_variable']
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.pos_sorted_unique = None
        self.pos_dif = None
        self.pos_lims = None
        self.amr_shape = None
        self.c_x = None
        self.c_y = None
        self.c_z = None
        self.c_xyz_mins = None
        self.c_xyz_difs = None
        self.c_grid_dims = None
        self.setFilledAMRShape()
        self.amr_prog_grid = self.fillAMRGrid(self.amr_prog_grid, self.cell_prog, 'progress_variable')
        self.amr_curv_grid = self.fillAMRGrid(self.amr_curv_grid, self.cell_curv, 'curvature')
        self.amr_gx_grid = self.fillAMRGrid(self.amr_gx_grid, self.cell_grads[0], 'grad x')
        self.amr_gy_grid = self.fillAMRGrid(self.amr_gy_grid, self.cell_grads[1], 'grad y')
        self.amr_gz_grid = self.fillAMRGrid(self.amr_gz_grid, self.cell_grads[2], 'grad z')

        print('making interpolators')
        interp_range = (self.x_range, self.y_range, self.z_range)
        self.prog_interp = RegularGridInterpolator(interp_range, self.amr_prog_grid)
        self.curv_interp = RegularGridInterpolator(interp_range, self.amr_curv_grid)
        self.gx_interp = RegularGridInterpolator(interp_range, self.amr_gx_grid)
        self.gy_interp = RegularGridInterpolator(interp_range, self.amr_gy_grid)
        self.gz_interp = RegularGridInterpolator(interp_range, self.amr_gz_grid)
        print('done!')

    def getGACoords(self):
        x = self.x_range[round(self.amr_shape[0] / 2)]
        y = self.y_range[round(self.amr_shape[1] / 2)]
        z = self.z_range[round(self.amr_shape[2] / 10)]
        return [x, y, z]

    @staticmethod
    def pointDist(p1, p2):
        return np.sqrt(sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

    def gradAscentPoint(self):
        amr_coords = self.getGACoords()
        xyz = self.cellPosToGridIndex(amr_coords)

        cur_prog = self.amr_prog_grid[xyz[0]][xyz[1]][xyz[2]]
        cur_curv = self.amr_curv_grid[xyz[0]][xyz[1]][xyz[2]]
        gx = [self.amr_gx_grid[xyz[0]][xyz[1]][xyz[2]]]
        gy = [self.amr_gy_grid[xyz[0]][xyz[1]][xyz[2]]]
        gz = [self.amr_gz_grid[xyz[0]][xyz[1]][xyz[2]]]

        r_arr = [0]
        prog_arr = [cur_prog]
        curv_arr = [cur_curv]

        last_point = amr_coords

        while cur_prog < 0.98:
            cur_point = GradAscent.perform_gA(last_point, gx, gy, gz, 1)
            r_arr.append(self.pointDist(cur_point, last_point) + r_arr[-1])
            cur_prog = self.prog_interp(cur_point)[0]
            cur_curv = self.curv_interp(cur_point)[0]
            # print(f'new point {cur_point} with prog {cur_prog} and curv {cur_curv}')
            prog_arr.append(cur_prog)
            curv_arr.append(cur_curv)
            gx = self.gx_interp(cur_point)
            gy = self.gy_interp(cur_point)
            gz = self.gz_interp(cur_point)
            last_point = cur_point

        print(f'r: {r_arr}')
        print(f'prog: {prog_arr}')
        print(f'curv: {curv_arr}')
        print(len(r_arr))
        print(len(prog_arr))
        print(len(curv_arr))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        H, x_edges, y_edges, bin_num = binned_statistic_2d(r_arr, prog_arr, values=curv_arr, statistic='mean', bins=[75, 75])
        H = np.ma.masked_invalid(H)
        XX, YY = np.meshgrid(x_edges, y_edges)
        p1 = ax.pcolormesh(XX, YY, H.T)
        cbar = fig.colorbar(p1, ax=ax, label='curvature')
        ax.set_xlabel('r')
        ax.set_ylabel('progress variable ')
        fig.tight_layout()
        plt.show()

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
        self.amr_shape = np.array([round((self.pos_lims[i][1] - self.pos_lims[i][0]) / self.pos_dif[i]) + 1 for i in range(3)], dtype=np.int32)

        self.amr_prog_grid = np.zeros(self.amr_shape)
        self.amr_curv_grid = np.zeros(self.amr_shape)
        self.amr_gx_grid = np.zeros(self.amr_shape)
        self.amr_gy_grid = np.zeros(self.amr_shape)
        self.amr_gz_grid = np.zeros(self.amr_shape)

        self.x_range = self.pos_sorted_unique[0]
        self.y_range = self.pos_sorted_unique[1]
        self.z_range = self.pos_sorted_unique[2]

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


if __name__ == '__main__':
    # CSVtoPKL('combGradCurv01929Data')
    file_name = 'res/combGradCurv01929Data.pkl'
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    print(amr_data.columns)
    print(f'Setting up grid ...')
    grid = AMRGrid(amr_data)
    grid.gradAscentPoint()
