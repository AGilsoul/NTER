import random

from scipy.signal import savgol_filter
from formatDAT import formatDAT
from ReadDat import ReadDat
import numpy as np
import matplotlib.pyplot as plt
from ProbDistFunc import pdf_2D, plot_against_2D
import pandas as pd
from matplotlib import animation

# formatCSV('res/01920Lev3.csv')
# formatCSV('res/01930Lev3.csv')
# CSVtoPKL('01920Lev3')
# CSVtoPKL('01930Lev3')


i_min = 1000
i_max = 2001


# basic vector operations
class VectorOps:
    @staticmethod
    def deriv(x, y, x1, y1):
        dy = x1 - x
        dx = y1 - y
        return dy / dx

    @staticmethod
    def point_dist(p0, p1):
        dx = p0['X'] - p1['X']
        dy = p0['Y'] - p1['Y']
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def tan_from_norm(p):
        return np.array([p[1], -p[0]])


# setup dat file, calculate curvature, radius, etc
def dat_setup(file_name, smooth):
    formatDAT(f'{file_name}.dat')
    t, data = ReadDat(f'{file_name}.dat')
    data['t'] = [t for _ in data['X']]
    data = data.sort_values('X')
    data = data.reset_index(drop=True)

    curv = []
    radius = []

    data['dx'] = (data['X'].shift(-1) - data['X'].shift(1))
    data['dy'] = (data['Y'].shift(-1) - data['Y'].shift(1))
    data['dnx'] = (data['progress_variable_gx'].shift(-1) - data['progress_variable_gx'].shift(1))
    data['dny'] = (data['progress_variable_gy'].shift(-1) - data['progress_variable_gy'].shift(1))
    data['curvature'] = -((data['dnx'] / data['dx']) + (data['dny'] / data['dx']))
    data['r'] = (1 / data['curvature'])
    if smooth:
        data['curvature'] = savgol_filter(data['curvature'], 20, 3)
        data['r'] = savgol_filter(data['r'], 20, 3)
        pd.to_pickle(data, f'{file_name}_smooth.pkl')
    else:
        pd.to_pickle(data, f'{file_name}.pkl')


# format all dat files
def format_all_dat(file_name, smooth):
    dat_setup(file_name, smooth)
    print('Done!')


def interp_helper(p1, p2, dx1, dx2, d12, quantity):
    return (p1[quantity] * dx1 + p2[quantity] * dx2) / d12


def interpolate_point(X, p1, p2, q, dt):
    p1_arr = np.array([p1['X'], p1['Y']])
    p2_arr = np.array([p2['X'], p2['Y']])
    d_x_p1 = X - p1_arr
    d_x_p1 = np.sqrt(np.dot(d_x_p1, d_x_p1))
    d_x_p2 = X - p2_arr
    d_x_p2 = np.sqrt(np.dot(d_x_p2, d_x_p2))
    d_p1_p2 = p1_arr - p2_arr
    d_p1_p2 = np.sqrt(np.dot(d_p1_p2, d_p1_p2))

    u = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'x_velocity')
    v = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'y_velocity')
    gx = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'progress_variable_gx')
    gy = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'progress_variable_gy')
    k = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'curvature')
    r = 1/k
    temp = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'temp')
    mix_frac = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'mixture_fraction')
    Y_OH = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'Y_OH')
    Y_H = interp_helper(p1, p2, d_x_p1, d_x_p2, d_p1_p2, 'Y_H')
    return {'X': X[0],
            'Y': X[1],
            'x_velocity': u,
            'y_velocity': v,
            'progress_variable_gx': gx,
            'progress_variable_gy': gy,
            'curvature': k,
            'r': r,
            'temp': temp,
            't': p1['t'],
            'dt': dt,
            'mixture_fraction': mix_frac,
            'Y_OH': Y_OH,
            'Y_H': Y_H,
            'q': q,
            'success': True
            }


def point_from_row(row, q, dt):
    u = row['x_velocity']
    v = row['y_velocity']
    gx = row['progress_variable_gx']
    gy = row['progress_variable_gy']
    k = row['curvature']
    r = 1 / k
    temp = row['temp']
    mix_frac = row['mixture_fraction']
    Y_OH = row['Y_OH']
    Y_H = row['Y_H']

    return {'X': row['X'],
            'Y': row['Y'],
            'x_velocity': u,
            'y_velocity': v,
            'progress_variable_gx': gx,
            'progress_variable_gy': gy,
            'curvature': k,
            'r': r,
            'temp': temp,
            't': row['t'],
            'dt': dt,
            'mixture_fraction': mix_frac,
            'Y_OH': Y_OH,
            'Y_H': Y_H,
            'q': q,
            'success': True
            }


# find the intersection of the direction vector found from point p and the line formed by points p1 and p2
def find_intersection(X, n, p1, p2):

    # [x1, y1] = [nx, ny]*t + [bx, by]

    m = n[1] / n[0]
    b0 = X[1] - m * X[0]

    # get current line
    slope = (p1['Y'] - p2['Y']) / (p1['X'] - p2['X'])
    b1 = p1['Y'] - slope * p1['X']

    # make sure lines aren't parallel
    if m == slope or np.isnan(slope) or np.isnan(m):
        return [-1, -1], -1, False

    # get x and y values of intersection
    # line is vertical
    if np.isinf(slope):
        x_intersection = p1['X']
        y_intersection = m * x_intersection + b0
    elif np.isinf(m):
        x_intersection = X[0]
        y_intersection = slope * x_intersection + b1
    elif slope == 0:
        y_intersection = p1['Y']
        x_intersection = (y_intersection - b0) / m
    elif m == 0:
        y_intersection = p1['Y']
        x_intersection = (y_intersection - b1) / slope
    else:
        x_intersection = (b1 - b0) / (m - slope)
        y_intersection = m * x_intersection + b0

    if not ((p1['X'] <= x_intersection <= p2['X'] or p2['X'] <= x_intersection <= p1['X']) and (p1['Y'] <= y_intersection <= p2['Y'] or p2['Y'] <= y_intersection <= p1['Y'])) or np.isnan(x_intersection):
        return [-1, -1], -1, False
    # get q value of vector
    q = (x_intersection - X[0]) / n[0]
    return np.array([x_intersection, y_intersection]), q, True


threshold = 0.001


# get point on next curve
def get_next_point(p, points, dt, last_q, first, index, plot=False, last_points=[]):
    inter_index = -1
    min_q = 0
    inter_coords = []
    # get current point position, and account for fluid flow
    X = np.array([p['X'] + p['x_velocity'] * dt, p['Y'] + p['y_velocity'] * dt])
    # get normal vector for direction
    n = np.array([p['progress_variable_gx'], p['progress_variable_gy']])
    n /= np.sqrt(np.dot(n, n))
    # for every point
    for i in range(len(points) - 1):
        cur_p1 = points.iloc[i]
        cur_p2 = points.iloc[i+1]
        inter, q, success = find_intersection(X, n, cur_p1, cur_p2)
        if not success:
            continue
        if inter_index == -1 or np.abs(min_q) >= np.abs(q):
            inter_index = i
            min_q = q
            inter_coords = inter

    if plot:
        print(min_q)
        plt.scatter(points['X'], points['Y'])
        plt.scatter(X[0], X[1])
        plt.quiver(X[0], X[1], n[0] * min_q, n[1] * min_q, angles='xy', scale_units='xy', scale=1)
        plt.show()

    if inter_index == -1:
        return {'success': False}

    res = interpolate_point(inter_coords, points.iloc[inter_index], points.iloc[inter_index + 1], min_q, dt)
    res['dy'] = res['X'] - p['X']
    '''
    print(f'minimum q: {min_q}')
    print(f'q flame speed: {min_q / dt}')
    fluid_diff = X - np.array([p['X'], p['Y']])
    fluid_disp = np.sqrt(np.dot(fluid_diff, fluid_diff))
    flame_diff = np.array([res['X'], res['Y']]) - X
    flame_disp = np.sqrt(np.dot(flame_diff, flame_diff))
    print(f'fluid displacement: {fluid_disp}')
    print(f'flame displacement: {flame_disp}')
    print(f'dt: {dt}')
    print(f'calculated flame speed: {flame_disp / dt}')
    plt.plot(points['X'], points['Y'], label='surface t+1')
    if len(last_points) != 0:
        plt.plot(last_points['X'], last_points['Y'], label='surface t')
    plt.xlim(0, 0.032)
    plt.ylim(0, 0.032)
    plt.scatter(X[0], X[1], label='fluid push')
    plt.scatter(p['X'], p['Y'], label='original')
    plt.quiver(p['X'], p['Y'], p['x_velocity'] * dt, p['y_velocity'] * dt, angles='xy', scale_units='xy', scale=1)
    plt.quiver(X[0], X[1], n[0] * min_q, n[1] * min_q, angles='xy', scale_units='xy', scale=1)
    plt.scatter(res['X'], res['Y'], label='end point')
    plt.legend()
    plt.show()
    # '''
    return res


def process_surfaces(reg=False, smooth=False):
    # process all files
    file_names = []
    print('Formatting all files ...')
    for i in range(i_min, i_max):
        file_name = f'res/2d_paths/pathIso{str(i).zfill(5)}'
        if reg:
            file_name += 'Reg'
        if smooth:
            file_name += '_smooth'
        file_names.append(file_name)
    print('All files formatted:')
    isoT0 = pd.read_pickle(f'{file_names[0]}.pkl')
    isoT0 = isoT0.dropna()
    isoT0 = isoT0.reset_index()
    indices = np.arange(100, len(isoT0) - 100, 5)
    num_points = len(indices)
    print(f'Evaluating at {num_points} points ...')
    p0 = [isoT0.iloc[indices[i]] for i in range(num_points)]

    all_points = [[] for _ in range(num_points)]
    skip_indices = []
    last_p0 = p0
    last_q = [0 for _ in range(num_points)]
    for i in range(i_min+1, i_max):
        print(i)
        isoT1 = pd.read_pickle(f'{file_names[i-i_min]}.pkl')
        print(skip_indices)
        print([indices[i] for i in skip_indices])
        for p in range(len(p0)):
            if p in skip_indices:
                continue
            cur_p0 = p0[p]
            isoT1_copy = isoT1.copy()
            isoT1_copy = isoT1_copy.dropna()
            isoT1_copy = isoT1_copy.reset_index()
            isoT1_copy['dist'] = isoT1_copy.apply(lambda row: VectorOps.point_dist(row, cur_p0), axis=1)
            isoT1_copy = isoT1_copy[isoT1_copy['dist'] < threshold]
            dt = 5e-6
            cur_p0 = get_next_point(cur_p0, isoT1_copy, dt, last_q[p], i == 1001, p, last_points=isoT0)
            if not cur_p0['success']:
                print('Failed!')
                skip_indices.append(p)
                break
            cur_p0['dr'] = cur_p0['r'] - last_p0[p]['r']
            all_points[p].append(cur_p0)
            last_p0[p] = p0[p]
            last_q[p] = cur_p0['q']
            p0[p] = cur_p0
        isoT0 = isoT1

    for p in range(len(all_points)):
        if p in skip_indices:
            continue
        path_data = pd.DataFrame.from_dict(all_points[p])
        path_data.dropna()
        if smooth:
            path_data.to_pickle(f'res/2d_paths/{p}PathData_smooth.pkl')
        else:
            path_data.to_pickle(f'res/2d_paths/{p}PathData.pkl')

    return num_points


def create_regular_grid(file_name, dim=100000, smooth=False):
    cols = ['X', 'Y', 'x_velocity', 'y_velocity', 'density', 'temp', 'HeatRelease', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_H', 'Y_O', 'Y_OH', 'Y_HO2', 'Y_H2O2', 'Y_N2', 'mixture_fraction', 'progress_variable', 'progress_variable_gx', 'progress_variable_gy', '||gradprogress_variable||', 'curvature', 'r', 't']

    if smooth:
        data = pd.read_pickle(f'{file_name}_smooth.pkl')
    else:
        data = pd.read_pickle(f'{file_name}.pkl')
    print(f'original len: {len(data)}')
    data = data.dropna()
    print(f'new len: {len(data)}')
    print()
    points = np.array(data['X'].values)
    grid_x = np.linspace(0, 0.032, dim)
    all_data = []
    k = 2
    for x in range(len(grid_x)):
        pt_data = [grid_x[x]]
        for i in range(1, len(cols) - 1):
            pt_data.append(np.interp(grid_x[x], points, data[cols[i]]))
        pt_data.append(data.iloc[0]['t'])
        all_data.append(pt_data)
    new_data = pd.DataFrame(all_data)
    new_data.columns = cols
    if smooth:
        pd.to_pickle(new_data, f'{file_name}Reg_smooth.pkl')
    else:
        pd.to_pickle(new_data, f'{file_name}Reg.pkl')
    print(new_data)


def regular_process(smooth=False):
    for i in range(1000, 2001):
        print(f'Making regular grid from file {i} ...')
        file_name = f'res/2d_paths/pathIso{str(i).zfill(5)}'
        create_regular_grid(file_name, smooth=smooth)
    print(f'Done!')


def process_dats(smooth=False):
    for i in range(1000, 2001):
        print(f'Making pkl from file {i} ...')
        file_name = f'res/2d_paths/pathIso{str(i).zfill(5)}'
        format_all_dat(file_name, smooth)
    print(f'Done!')


# p_list = [1, 2, 4, 5]
p_list = []


def update_fig(t, point_paths, ax_an1, ax_an2, x_axis, y_axis, num_points):
    step = int(t) % 1000
    print(t)
    # print(data)
    global surfaces
    cur_surface = surfaces[step]
    ax_an1.cla()
    ax_an2.cla()
    # mag_vel = np.sqrt((cur_point['x_velocity'] * cur_point['dt'])**2 + (cur_point['y_velocity'] * cur_point['dt'])**2)
    ax_an1.set_xlim(0, 0.032)
    ax_an1.set_ylim(0, 0.032)
    scale = 1000
    
    # d = mag_vel * 100
    # ax_an1.set_xlim(cur_point['X'] - d, cur_point['X'] + d)
    # ax_an1.set_ylim(cur_point['Y'] - d, cur_point['Y'] + d)
    ax_an1.set_xlabel('X')
    ax_an1.set_ylabel('Y')
    # n = np.array([cur_point['progress_variable_gx'] * cur_point['q'], cur_point['progress_variable_gy'] * cur_point['q']])
    # u = np.array([cur_point['x_velocity'] * cur_point['dt'], cur_point['y_velocity'] * cur_point['dt']])
    ax_an1.plot(cur_surface['X'], cur_surface['Y'])
    ax_an2.set_xlabel(x_axis)
    ax_an2.set_ylabel(y_axis)
    for p in range(num_points):
        if p not in p_list:
            if step <= len(point_paths[p]):
                cur_point = point_paths[p].iloc[step]
                # print(cur_point)
                ax_an1.scatter(cur_point['X'], cur_point['Y'], s=5)
                ax_an2.plot(point_paths[p][x_axis], point_paths[p][y_axis])
                ax_an2.scatter(cur_point[x_axis], cur_point[y_axis], s=5)
    # ax_an2.set_xscale('log')
        # ax_an1.quiver(cur_point['X'], cur_point['Y'], n[0], n[1], angles='xy', scale_units='xy', scale=1)
    # ax_an1.quiver(cur_point['X'], cur_point['Y'], u[0] * scale, u[1] * scale, angles='xy', scale_units='xy', scale=1)
    # ax_an1.quiver(cur_point['X'] + u[0], cur_point['Y'] + u[1], n[0] * scale, n[1] * scale, angles='xy', scale_units='xy', scale=1)

    return ax_an1, ax_an2


surfaces = []


def graph_data(x_axis, y_axis, num_points=10, reg=False, smooth=False, animating=False):
    print(f'reading surfaces...')
    global surfaces
    if animating:
        for i in range(i_min, i_max):
            surfaces.append(pd.read_pickle(f'res/2d_paths/pathIso0{i}Reg_smooth.pkl'))
    point_paths = []
    print(f'reading point data...')
    for p in range(num_points):
        if smooth:
            data = pd.read_pickle(f'res/2d_paths/{p}PathData_smooth.pkl')
        else:
            data = pd.read_pickle(f'res/2d_paths/{p}PathData.pkl')
        data['k_hat'] = savgol_filter(data['curvature'], 50, 3)
        data['dr_hat'] = (1/data['k_hat'] - 1/data['k_hat'].shift(1))
        # data['dr_hat'] = savgol_filter(data['dr'], 50, 3)
        data['drdt'] = data['dr_hat'] / data['dt']
        data['kdrdt'] = data['k_hat'] * data['dr_hat'] / data['dt']
        # data['kdrdt'] = savgol_filter(data['kdrdt'], 50, 3)
        data = data.dropna()
        point_paths.append(data)
    print('Done reading data')
    fig = plt.figure()
    if animating:
        ax_an1 = fig.add_subplot(1, 2, 1)
        ax_an2 = fig.add_subplot(1, 2, 2)
        # fig.tight_layout()
        print(f'animating ...')
        anim = animation.FuncAnimation(fig, update_fig, fargs=(point_paths, ax_an1, ax_an2, x_axis, y_axis, num_points),
                                       interval=1, frames=i_max - i_min - 3, repeat=False, blit=False)
        # plt.show()
        # writervideo = animation.FFMpegWriter(fps=60)
        # anim.save('test_anim.mp4', writer=writervideo)
        anim.save(f'{num_points}{y_axis}Plot.gif', writer='imagemagick', fps=60)
        plt.close()
    else:
        # each index contains all points for cur timestep
        '''
        x_coords = np.array([point_paths[p]['X'].values for p in range(num_points)])
        y_coords = np.array([point_paths[p]['Y'].values for p in range(num_points)])
        x_vals_per_point = np.array([point_paths[p][x_axis].values for p in range(num_points)])
        y_vals_per_point = np.array([point_paths[p][y_axis] for p in range(num_points)])
        t_vals_per_point = np.array([point_paths[p]['t'] for p in range(num_points)])
        other_vals_per_point = np.array([point_paths[p]['q'] / point_paths[p]['dt'] for p in range(num_points)])
        x_vals_per_time = x_vals_per_point.T
        y_vals_per_time = y_vals_per_point.T
        ax_an1 = fig.add_subplot(1, 2, 1)
        ax_an2 = fig.add_subplot(1, 2, 2)
        ax_an1.cla()
        ax_an2.cla()
        # n = np.array([cur_point['progress_variable_gx'] * cur_point['q'], cur_point['progress_variable_gy'] * cur_point['q']])
        # u = np.array([cur_point['x_velocity'] * cur_point['dt'], cur_point['y_velocity'] * cur_point['dt']])
        ax_an1.set_xlabel(x_axis)
        ax_an1.set_ylabel(y_axis)
        ax_an2.set_xlabel('t')
        ax_an2.set_ylabel(x_axis)

        index = 0
        ax_an1.scatter(x_vals_per_point[index], y_vals_per_point[index])
        # ax_an2.plot(t_vals_per_point[index], x_vals_per_point[index])
        # '''
        '''
        all_x_vals = []
        all_y_vals = []
        all_t_vals = []
        all_speed_vals = []
        indices = []
        for p in range(len(point_paths)):
            for i in range(len(point_paths[p][x_axis].values)):
                valx = point_paths[p][x_axis].values[i]
                valy = point_paths[p][y_axis].values[i]
                valt = point_paths[p]['t'].values[i]
                val_speed = point_paths[p]['q'].values[i] / point_paths[p]['dt'].values[i]
                if True:
                    all_x_vals.append(valx)
                    all_y_vals.append(valy)
                    all_t_vals.append(valt)
                    all_speed_vals.append(val_speed)
        all_x_vals = np.array(all_x_vals)
        all_y_vals = np.array(all_y_vals)
        all_t_vals = np.array(all_t_vals)
        all_speed_vals = np.array(all_speed_vals)

        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.set_xscale('log')
        pdf_2D(fig, ax, all_x_vals, all_y_vals, x_axis, y_axis, dpi=100)
        # '''
        fig.tight_layout()
        plt.show()
    '''
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    # ax1.scatter(data['t'], data['curvature'], label='Original', s=1)
    ax1.plot(data['t'], data['kdrdt'], label='Smooth')
    ax1.legend()
    ax1.set_xlabel('t')
    ax1.set_ylabel('kdrdt')
    ax1.set_title('kdrdt over time')
    ax2.scatter(data['X'], data['Y'], c=plt.cm.hot(data['t'] / max(data['t'])), edgecolor='none')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('position over time')
    ax3.plot(data['t'], data['X'])
    ax3.set_xlabel('t')
    ax3.set_ylabel('X')
    ax3.set_title('X position over time')
    ax4.plot(data['t'], data['Y'])
    ax4.set_xlabel('t')
    ax4.set_ylabel('Y')
    ax4.set_title('Y position over time')
    plt.tight_layout()
    plt.show()
    '''


def test_stuff():
    data = pd.read_pickle(f'res/2d_paths/pathIso00435Reg.pkl')
    path_data = pd.read_pickle(f'res/2d_paths/600PathData.pkl')
    path_data = path_data[path_data['t'] == 0.00243757]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'{data["t"][0]}')
    plot_against_2D(fig, ax, data['X'], data['Y'], data['curvature'], 'X', 'Y', 'curvature', bins=[10000, 10000])
    plt.plot(data['X'], data['Y'])
    plt.scatter(path_data['X'], path_data['Y'])
    plt.show()


# do more evenly spaced points, a single point, a bunch of points with pdf
# plot quantities against k dr/dt


def main():
    # new_curvature()
    # process_dats(smooth=False)
    # regular_process(smooth=True)
    num_points = 314
    # res = process_surfaces(reg=False, smooth=True)
    # print(res)
    graph_data('t', 'Y_OH', num_points=num_points, reg=False, smooth=True, animating=True)
    # test_stuff()
    return


if __name__ == '__main__':
    main()