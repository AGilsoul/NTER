from ctypes import cdll, c_int, c_float, POINTER

import matplotlib
import pandas as pd
import numpy as np
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import sys

from csvToPkl import CSVtoPKL

FloatPtr = POINTER(c_float)
IntPtr = POINTER(c_int)

# lib_path = 'lib/GradAscent.so'  # linux
lib_path = 'lib/GradAscent.dll'  # windows
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

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)


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
        self.cell_temp = self.amr_data['temp']
        self.cell_validity = [1.0 for _ in range(len(self.amr_data))]
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
        self.amr_temp_grid = self.fillAMRGrid(self.amr_temp_grid, self.cell_temp, 'temp')
        self.amr_gx_grid = self.fillAMRGrid(self.amr_gx_grid, self.cell_grads[0], 'grad x')
        self.amr_gy_grid = self.fillAMRGrid(self.amr_gy_grid, self.cell_grads[1], 'grad y')
        self.amr_gz_grid = self.fillAMRGrid(self.amr_gz_grid, self.cell_grads[2], 'grad z')
        self.amr_validity_grid = self.fillAMRGrid(self.amr_validity_grid, self.cell_validity, 'cell validity')

        print('making interpolators')
        interp_range = (self.x_range, self.y_range, self.z_range)
        self.prog_interp = RegularGridInterpolator(interp_range, self.amr_prog_grid)
        self.curv_interp = RegularGridInterpolator(interp_range, self.amr_curv_grid)
        self.temp_interp = RegularGridInterpolator(interp_range, self.amr_temp_grid)
        self.gx_interp = RegularGridInterpolator(interp_range, self.amr_gx_grid)
        self.gy_interp = RegularGridInterpolator(interp_range, self.amr_gy_grid)
        self.gz_interp = RegularGridInterpolator(interp_range, self.amr_gz_grid)
        self.valid_interp = RegularGridInterpolator(interp_range, self.amr_validity_grid)
        print('done!')

    def getGACoords(self):
        quarter_lim = self.x_range[round(self.amr_shape[0]/4)]
        half_lim = self.x_range[round(self.amr_shape[0]/2)]
        x = np.linspace(quarter_lim, half_lim + quarter_lim, 10)
        y = np.linspace(quarter_lim, half_lim + quarter_lim, 10)
        z = self.z_range[2]

        g = np.meshgrid(x, y, [z])
        points = np.vstack(list(map(np.ravel, g))).T
        print(points.shape)

        # x = self.x_range[round(self.amr_shape[0] / 2)]
        # y = self.y_range[round(self.amr_shape[1] / 2)]
        # return [x, y, z]
        return points

    def validPoint(self, p):
        x_in_grid = self.pos_lims[0][1] > p[0] >= self.pos_lims[0][0]
        y_in_grid = self.pos_lims[1][1] > p[1] >= self.pos_lims[1][0]
        z_in_grid = self.pos_lims[2][1] > p[2] >= self.pos_lims[2][0]
        valid_cell = self.valid_interp(p) != 0.0
        return x_in_grid and y_in_grid and z_in_grid and valid_cell

    @staticmethod
    def pointDist(p1, p2):
        return np.sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]))

    def gradAscentPoint(self, amr_coords):
        amr_coords = np.array(amr_coords, dtype=np.float32)
        num_pts = amr_coords.shape[0]
        print(f'Evaluating {num_pts} points ...')
        xyz = [self.cellPosToGridIndex(i) for i in amr_coords]
        cur_progs = [[self.amr_prog_grid[xyz[i][0]][xyz[i][1]][xyz[i][2]]] for i in range(num_pts)]
        cur_curvs = [[self.amr_curv_grid[xyz[i][0]][xyz[i][1]][xyz[i][2]]] for i in range(num_pts)]
        cur_temps = [[self.amr_temp_grid[xyz[i][0]][xyz[i][1]][xyz[i][2]]] for i in range(num_pts)]

        r_arr = [[0] for _ in amr_coords]
        path_arr = [[list(i)] for i in amr_coords]
        prog_arr = cur_progs
        curv_arr = cur_curvs
        temp_arr = cur_temps

        valid_indices = np.arange(0, num_pts)

        last_points = amr_coords

        last_num_pts = -1

        while True:
            p_to_remove = []
            for p in range(len(last_points)):
                if not self.validPoint(last_points[p]) or prog_arr[p][-1] > 0.99:
                    p_to_remove.append(p)
            valid_indices = np.delete(valid_indices, p_to_remove)
            # print(f'indices to remove: {p_to_remove}')
            last_points = np.delete(last_points, p_to_remove, axis=0)
            num_pts = last_points.shape[0]
            if len(last_points) == 0:
                break
            if last_num_pts != num_pts:
                print(f'{num_pts} points left!')
                print(f'valid indices: {valid_indices}')
            gx = np.array([self.gx_interp(p) for p in last_points], dtype=np.float32)
            gy = np.array([self.gy_interp(p) for p in last_points], dtype=np.float32)
            gz = np.array([self.gz_interp(p) for p in last_points], dtype=np.float32)

            cur_points = GradAscent.perform_gA(np.array(last_points, dtype=np.float32).flatten(), gx.flatten(), gy.flatten(), gz.flatten(), num_pts).reshape(last_points.shape)
            for p in range(len(cur_points)):
                index = valid_indices[p]
                r_arr[index].append(self.pointDist(cur_points[p], last_points[p]) + r_arr[index][-1])
                prog_arr[index].append(self.prog_interp(cur_points[p])[0])
                # curv_arr[index].append(self.curv_interp(cur_points[p])[0])
                # path_arr[index].append(list(cur_points[p]))
                temp_arr[index].append(self.temp_interp(cur_points[p])[0])

            last_points = cur_points
            last_num_pts = num_pts

        # print(path_arr)
        num_pts = amr_coords.shape[0]

        point_to_plot = 0

        fig = plt.figure()
        # self.pdf_2D(fig, 1, 1, 1, prog_arr[point_to_plot], temp_arr[point_to_plot], 'Progress Variable C', 'Temperature T')
        plt.plot(prog_arr[point_to_plot], temp_arr[point_to_plot])
        plt.xlabel('Progress Variable C')
        plt.ylabel('Temperature T')
        # H, x_edges, y_edges, bin_num = binned_statistic_2d(prog_arr[point_to_plot], values=curv_arr[p], statistic='mean', bins=[100, 100])
        # H = np.ma.masked_invalid(H)
        # XX, YY = np.meshgrid(x_edges, y_edges)
        # p1 = ax.pcolormesh(XX, YY, H.T)
        # cbar = fig.colorbar(p1, ax=ax, label='curvature')
        # ax.set_xlabel('r')
        # ax.set_ylabel('progress variable ')

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
        self.amr_temp_grid = np.zeros(self.amr_shape)
        self.amr_gx_grid = np.zeros(self.amr_shape)
        self.amr_gy_grid = np.zeros(self.amr_shape)
        self.amr_gz_grid = np.zeros(self.amr_shape)
        self.amr_validity_grid = np.zeros(self.amr_shape)

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

    @staticmethod
    def plot_against_2D(fig, r, c, i, xd, yd, z, xlabel, ylabel, zlabel):
        z = [d if d >= 0 else 0 for d in z]
        ax = fig.add_subplot(r, c, i)
        H, x_edges, y_edges, bin_num = binned_statistic_2d(xd, yd, values=z, statistic='mean', bins=[500, 500])
        H = np.ma.masked_invalid(H)
        XX, YY = np.meshgrid(x_edges, y_edges)
        p1 = ax.pcolormesh(XX, YY, H.T)
        cbar = fig.colorbar(p1, ax=ax, label=zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    @staticmethod
    def pdf_2D(fig, r, c, i, xd, yd, xlabel, ylabel):
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
        ax = fig.add_subplot(r, c, i, projection='scatter_density')
        density = ax.scatter_density(xd, yd, cmap=white_viridis, dpi=70,
                                     norm=matplotlib.colors.SymLogNorm(linthresh=0.03))
        fig.colorbar(density, label='Number of points per pixel')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return


if __name__ == '__main__':
    # CSVtoPKL('combGradCurv01929Data')
    file_name = 'res/combGradCurv01929Data.pkl'
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    print(amr_data.columns)
    print(f'Setting up grid ...')
    grid = AMRGrid(amr_data)
    coords = grid.getGACoords()
    grid.gradAscentPoint(coords)
