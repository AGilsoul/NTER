from ctypes import cdll, c_int, c_float, POINTER, c_bool

import matplotlib
import pandas as pd
import numpy as np
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d
import sys
import itertools

from csvToPkl import CSVtoPKL
from src.Cantera import compute1DFlame

FloatPtr = POINTER(c_float)
IntPtr = POINTER(c_int)

# lib_path = 'lib/GradAscent.so'  # linux
lib_path = 'lib/GradAscent.dll'  # windows
GradAscentLib = cdll.LoadLibrary(lib_path)
gradAscent = GradAscentLib.getNextPoints

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

gradAscent.argtypes = [FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               FloatPtr,
                               IntPtr,
                               c_int
                    ]
gradAscent.restype = FloatPtr


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
    def perform_gA(x, y, z, gx, gy, gz, c_xyz_mins, c_xyz_difs, c_grid_dims, num_pts):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)

        gx = np.array(gx, dtype=np.float32)
        gy = np.array(gy, dtype=np.float32)
        gz = np.array(gz, dtype=np.float32)

        c_x = x.ctypes.data_as(FloatPtr)
        c_y = y.ctypes.data_as(FloatPtr)
        c_z = z.ctypes.data_as(FloatPtr)
        c_gx = gx.ctypes.data_as(FloatPtr)
        c_gy = gy.ctypes.data_as(FloatPtr)
        c_gz = gz.ctypes.data_as(FloatPtr)

        res_pts = gradAscent(c_x, c_y, c_z, c_gx, c_gy, c_gz, c_xyz_mins, c_xyz_difs, c_grid_dims, num_pts)
        res = np.ctypeslib.as_array(res_pts, shape=(num_pts*3,))
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
        self.cell_curv = self.amr_data['MeanCurvature_progress_variable']
        self.cell_temp = self.amr_data['temp']
        self.cell_mix = self.amr_data['mixture_fraction']
        self.cell_strainrate = self.amr_data['StrainRate_progress_variable']
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
        self.amr_strainrate_grid = self.fillAMRGrid(self.amr_strainrate_grid, self.cell_strainrate, 'strain rate')
        self.amr_curv_grid = self.fillAMRGrid(self.amr_curv_grid, self.cell_curv, 'curvature')
        self.amr_temp_grid = self.fillAMRGrid(self.amr_temp_grid, self.cell_temp, 'temp')
        self.amr_mix_grid = self.fillAMRGrid(self.amr_mix_grid, self.cell_mix, 'mixture fraction')
        self.amr_gx_grid = self.fillAMRGrid(self.amr_gx_grid, self.cell_grads[0], 'grad x')
        self.amr_gy_grid = self.fillAMRGrid(self.amr_gy_grid, self.cell_grads[1], 'grad y')
        self.amr_gz_grid = self.fillAMRGrid(self.amr_gz_grid, self.cell_grads[2], 'grad z')
        self.amr_validity_grid = self.fillAMRGrid(self.amr_validity_grid, self.cell_validity, 'cell validity')
        print('making interpolators ...')
        interp_range = (self.x_range, self.y_range, self.z_range)
        self.prog_interp = RegularGridInterpolator(interp_range, self.amr_prog_grid)
        self.strainrate_interp = RegularGridInterpolator(interp_range, self.amr_strainrate_grid)
        self.curv_interp = RegularGridInterpolator(interp_range, self.amr_curv_grid)
        self.temp_interp = RegularGridInterpolator(interp_range, self.amr_temp_grid)
        self.mix_interp = RegularGridInterpolator(interp_range, self.amr_mix_grid)
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
        return points

    def validPoint(self, p):
        x_in_grid = self.pos_lims[0][1] > p[0] >= self.pos_lims[0][0]
        y_in_grid = self.pos_lims[1][1] > p[1] >= self.pos_lims[1][0]
        z_in_grid = self.pos_lims[2][1] > p[2] >= self.pos_lims[2][0]
        if x_in_grid and y_in_grid and z_in_grid:
            return self.valid_interp(p) != 0.0
        return False

    @staticmethod
    def pointDist(p1, p2):
        return np.sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]))

    def nearestPointBelowProg(self, p, prog_val, search_thresh=0.0025):
        # valid_indices = [i for i in range(len(self.cell_prog)) if self.cell_prog[i] <= prog_val]
        valid_indices = self.amr_data.index[(self.amr_data['progress_variable'] <= prog_val) & (abs(self.amr_data['CellCenters:0'] - p[0]) < search_thresh) & (abs(self.amr_data['CellCenters:1'] - p[1]) < search_thresh) & (abs(self.amr_data['CellCenters:2'] - p[2]) < search_thresh)].tolist()
        all_points = np.array(self.cell_pos).T
        valid_points = [all_points[i] for i in valid_indices]
        valid_points = list(filter(lambda x: self.validPoint(x), valid_points))

        distances = [self.pointDist(p, valid_p) for valid_p in valid_points]

        return valid_points[distances.index(min(distances))]

    def gradAscentPoint(self, amr_coords, c_lim=0.99):
        amr_coords = np.array(amr_coords, dtype=np.float32)
        num_pts = amr_coords.shape[0]
        cur_progs = [[self.prog_interp(p)[0]] for p in amr_coords]
        cur_curvs = [[self.curv_interp(p)[0]] for p in amr_coords]
        cur_temps = [[self.temp_interp(p)[0]] for p in amr_coords]

        r_arr = [[0] for _ in amr_coords]
        prog_arr = cur_progs
        curv_arr = cur_curvs
        path_arr = [[list(i)] for i in amr_coords]

        valid_indices = np.arange(0, num_pts)

        last_points = amr_coords

        last_num_pts = -1

        iters = 0

        while True:
            if iters % 1000 == 0:
                print(f'cur progress: {prog_arr[0][-1]}, {prog_arr[1][-1]}, {prog_arr[2][-1]}')
                print(f'cur curv: {curv_arr[0][-1]}, {curv_arr[1][-1]}, {curv_arr[2][-1]}')

            p_to_remove = []
            for p in range(len(last_points)):
                if not self.validPoint(last_points[p]) or prog_arr[valid_indices[p]][-1] > c_lim:
                    p_to_remove.append(p)
            valid_indices = np.delete(valid_indices, p_to_remove)
            last_points = np.delete(last_points, p_to_remove, axis=0)
            num_pts = last_points.shape[0]
            if num_pts == 0:
                break
            if num_pts != last_num_pts:
                print(f'{num_pts} points left!')
                print(f'valid indices: {valid_indices}')
            gx = np.array([self.gx_interp(p) for p in last_points], dtype=np.float32)
            gy = np.array([self.gy_interp(p) for p in last_points], dtype=np.float32)
            gz = np.array([self.gz_interp(p) for p in last_points], dtype=np.float32)
            cur_points = GradAscent.perform_gA(last_points.T[0], last_points.T[1], last_points.T[2], gx.flatten(), gy.flatten(), gz.flatten(), self.c_xyz_mins, self.c_xyz_difs, self.c_grid_dims, num_pts).reshape(last_points.shape)
            for p in range(len(cur_points)):
                try:
                    index = valid_indices[p]
                    prog = self.prog_interp(cur_points[p])[0]
                    curv = self.curv_interp(cur_points[p])[0]
                    r_arr[index].append(self.pointDist(cur_points[p], last_points[p]) + r_arr[index][-1])
                    prog_arr[index].append(prog)
                    curv_arr[index].append(curv)
                    path_arr[index].append(cur_points[p])
                except ValueError:
                    index = valid_indices[p]
                    r_arr[index].pop()
                    prog_arr[index].pop()
                    curv_arr[index].pop()
                    path_arr[index].pop()
                    print(f'point {cur_points[p]} out of range')

            last_points = cur_points
            last_num_pts = num_pts
            iters += 1

        return path_arr, r_arr, prog_arr

    def interpolateAttr(self, path, attr):
        interp = None
        if attr == 'temp':
            interp = self.temp_interp
        elif attr == 'curvature':
            interp = self.curv_interp
        elif attr == 'strainrate':
            interp = self.strainrate_interp
        else:
            interp = self.mix_interp

        attr_vals = []
        for p in path:
            attr_vals.append(interp(p)[0])
        return attr_vals

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
        shape = grid_to_fill.shape
        grid_to_fill = np.ctypeslib.as_array(res_pts, shape=(grid_size,))
        grid_to_fill = grid_to_fill.reshape(shape, order='F')
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
        self.amr_strainrate_grid = np.zeros(self.amr_shape)
        self.amr_curv_grid = np.zeros(self.amr_shape)
        self.amr_temp_grid = np.zeros(self.amr_shape)
        self.amr_mix_grid = np.zeros(self.amr_shape)
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
    def movingAverage(data, window_size):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    @staticmethod
    def plot_against_2D(fig, r, c, i, xd, yd, z, xlabel, ylabel, zlabel, overlay_x=None, overlay_y=None, overlay_label=''):
        # z = [d if d >= 0 else 0 for d in z]
        ax = fig.add_subplot(r, c, i)
        H, x_edges, y_edges, bin_num = binned_statistic_2d(xd, yd, values=z, statistic='mean', bins=[500, 500])
        print(H)
        H_masked = np.ma.masked_invalid(H).copy()
        XX, YY = np.meshgrid(x_edges, y_edges)
        p1 = ax.pcolormesh(XX, YY, H_masked.T)
        cbar = fig.colorbar(p1, ax=ax, label=zlabel)
        if overlay_x is not None and overlay_y is not None:
            H_t, x_edges_t, y_edges_t, bin_num_t = binned_statistic_2d(xd, yd, values=yd, statistic='mean',
                                                                       bins=[1000, 1000])
            print(x_edges_t)
            print(y_edges_t)
            y_av = []
            for col in range(len(x_edges_t)-1):
                y_av.append(np.nansum(H_t[col])/np.count_nonzero(~np.isnan(H_t[col])))

            print(y_av)
            y_av = savgol_filter(y_av, 100, 3)
            # x_av = AMRGrid.movingAverage(x_edges, 10)
            # print(x_av)
            # y_av = AMRGrid.movingAverage(y_edges, 10)
            # print(y_av)
            ax.plot(x_edges_t[1:], y_av, label='Average', linewidth=2, color='black')
            ax.plot(overlay_x, overlay_y, label=overlay_label, color='r', linewidth=2, linestyle='dashed')
            ax.axhline(y=1428.5, color='black', label='AFT', linewidth=2, linestyle='dashed')
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    @staticmethod
    def plot_against_2D_mult(fig, r, c, i, xd, yd, z, xlabel, ylabel, zlabel):
        ax = fig.add_subplot(r, c, i)
        for x in range(len(xd)):
            H, x_edges, y_edges, bin_num = binned_statistic_2d(xd[x], yd[x], values=z[x], statistic='mean', bins=[500, 500])
            H = np.ma.masked_invalid(H)
            XX, YY = np.meshgrid(x_edges, y_edges)
            p1 = ax.pcolormesh(XX, YY, H.T)
            cbar = fig.colorbar(p1, ax=ax, label=zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    @staticmethod
    def pdf_2D(fig, r, c, i, xd, yd, xlabel, ylabel, overlay_x=None, overlay_y=None, overlay_label=''):
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

        if overlay_x is not None and overlay_y is not None:
            ax.plot(overlay_x, overlay_y, label=overlay_label, color='r', linestyle='dashed')
            ax.axhline(y=1428.5, color='black', label='AFT', linestyle='dashed')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return


def PresFlamePaths3D():
    # CSVtoPKL('combGradCurv01930Data')
    file_name = 'res/combGradCurv01930Data.pkl'
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    print(amr_data.columns)
    print(f'Setting up grid ...')
    grid = AMRGrid(amr_data)
    high_flux_pt = [0.01725, 0.01875, 0.01275]
    # med_flux_pt = [0.0224, 0.0136293, 0.0119005]
    med_flux_pt = [0.0261804, 0.0289737, 0.0146741]
    low_flux_pt = [0.015644, 0.020422, 0.0157]
    # low_flux_pt = [0.0155955, 0.020262, 0.0160572]

    coords = [high_flux_pt,
              med_flux_pt,
              low_flux_pt]
    path, r, prog = grid.gradAscentPoint(coords, c_lim=0.9999)
    print('done with gradient path, interpolating...')
    high_flux_path, med_flux_path, low_flux_path = path[0], path[1], path[2]
    high_flux_r, med_flux_r, low_flux_r = r[0], r[1], r[2]
    print('pre interpolate lens')
    print(f'path 0 len: {len(high_flux_path)}')
    print(f'path 1 len: {len(med_flux_path)}')
    print(f'path 2 len: {len(low_flux_path)}')
    print('temp...')
    high_flux_temp = grid.interpolateAttr(high_flux_path, 'temp')
    med_flux_temp = grid.interpolateAttr(med_flux_path, 'temp')
    low_flux_temp = grid.interpolateAttr(low_flux_path, 'temp')
    print('mix...')
    high_flux_mix = grid.interpolateAttr(high_flux_path, '')
    med_flux_mix = grid.interpolateAttr(med_flux_path, '')
    low_flux_mix = grid.interpolateAttr(low_flux_path, '')
    print('curv...')
    high_flux_curv = grid.interpolateAttr(high_flux_path, 'curvature')
    med_flux_curv = grid.interpolateAttr(med_flux_path, 'curvature')
    low_flux_curv = grid.interpolateAttr(low_flux_path, 'curvature')
    print('strainrate...')
    high_flux_strainrate= grid.interpolateAttr(high_flux_path, 'strainrate')
    med_flux_strainrate = grid.interpolateAttr(med_flux_path, 'strainrate')
    low_flux_strainrate = grid.interpolateAttr(low_flux_path, 'strainrate')

    print('post interpolate lens')
    print(f'path 0 len: {len(high_flux_path)}')
    print(f'path 1 len: {len(med_flux_path)}')
    print(f'path 2 len: {len(low_flux_path)}')

    path_flat = list(itertools.chain(high_flux_path, med_flux_path, low_flux_path))
    r_flat = list(itertools.chain(high_flux_r, med_flux_r, low_flux_r))
    prog_flat = list(itertools.chain(prog[0], prog[1], prog[2]))
    temp_flat = list(itertools.chain(high_flux_temp, med_flux_temp, low_flux_temp))
    mix_flat = list(itertools.chain(high_flux_mix, med_flux_mix, low_flux_mix))
    curv_flat = list(itertools.chain(high_flux_curv, med_flux_curv, low_flux_curv))
    strainrate_flat = list(itertools.chain(high_flux_strainrate, med_flux_strainrate, low_flux_strainrate))

    print('post extension lens')
    print(f'path 0 len: {len(high_flux_path)}')
    print(f'path 1 len: {len(med_flux_path)}')
    print(f'path 2 len: {len(low_flux_path)}')

    flux_values = []
    for _ in high_flux_path:
        flux_values.append('hf')
    for _ in med_flux_path:
        flux_values.append('mf')
    for _ in low_flux_path:
        flux_values.append('lf')
    print(f'num flux vals: {len(flux_values)}')
    print(f'length of path: {len(path_flat)}')
    df = pd.DataFrame(list(zip(path_flat, r_flat, flux_values, prog_flat, temp_flat, mix_flat, curv_flat, strainrate_flat)),
                      columns=['path', 'r', 'flux', 'prog', 'temp', 'mix', 'curv', 'strainrate'])
    pd.to_pickle(df, 'res/FlamePaths.pkl')

    fig = plt.figure()
    grid.plot_against_2D(fig, 1, 1, 1, prog_flat, temp_flat, mix_flat, 'progress variable c', 'temperature T (k)', 'mixture fraction Z')
    plt.show()


if __name__ == '__main__':
    # PresFlamePaths3D()

    file_name = 'res/combGradCurv01930Data.pkl'
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    print(amr_data.columns)
    fig = plt.figure(figsize=(15, 15))
    matplotlib.rcParams.update({'font.size': 22})
    cantera_temp, cantera_c, cantera_z, grid = compute1DFlame()
    # AMRGrid.pdf_2D(fig, 1, 1, 1, amr_data['progress_variable'], amr_data['temp'], 'Progress Variable c', 'Temperature T (K)', cantera_c, cantera_temp, 'Cantera 1D')
    AMRGrid.plot_against_2D(fig, 1, 1, 1, amr_data['progress_variable'], amr_data['temp'], amr_data['MeanCurvature_progress_variable'], 'Progress Variable c', 'Temperature T (K)', 'Curvature K (1/m)', overlay_x=cantera_c, overlay_y=cantera_temp, overlay_label='Cantera 1D')
    plt.legend(loc='lower right')
    plt.show()
    # amr_data = amr_data.sample(10000)
    #iso_surface = amr_data[(amr_data['progress_variable'] <= 0.81) & (amr_data['progress_variable'] >= 0.79)]
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(iso_surface['CellCenters:0'], iso_surface['CellCenters:1'], iso_surface['CellCenters:2'])
    #ax.set_xlim(0, 0.032)
    #ax.set_ylim(0, 0.032)
    #ax.set_zlim(0, 0.032)
    #plt.show()
