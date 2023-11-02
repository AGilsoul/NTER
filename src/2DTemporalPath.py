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


def interp_helper(p1, p2, dx1, dx2, denom, quantity):
    return (p1[quantity] * dx1 + p2[quantity] * dx2) / denom


def interpolate_point(X, p1, p2, q, dt):
    d_x_p1 = X - np.array([p1['X'], p1['Y']])
    d_x_p1 = np.dot(d_x_p1, d_x_p1)
    d_x_p2 = X - np.array([p2['X'], p2['Y']])
    d_x_p2 = np.dot(d_x_p2, d_x_p2)
    denom = d_x_p1 + d_x_p2

    u = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'x_velocity')
    v = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'y_velocity')
    gx = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'progress_variable_gx')
    gy = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'progress_variable_gy')
    k = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'curvature')
    r = 1/k
    temp = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'temp')
    mix_frac = interp_helper(p1, p2, d_x_p1, d_x_p2, denom, 'mixture_fraction')

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
            'q': q,
            'success': True
            }


# find the intersection of the direction vector found from point p and the line formed by points p1 and p2
def find_intersection(X, n, b0, p1, p2):
    m = n[1] / n[0]

    # get slope of current line
    slope = VectorOps.deriv(p1['Y'], p1['X'], p2['Y'], p2['X'])
    b1 = p1['Y'] - slope * p1['X']

    # make sure lines aren't parallel
    if m == slope:
        return [-1, -1], -1, False

    # get x and y values of intersection
    x_intersection = (b1 - b0) / (m - slope)
    y_intersection = m * x_intersection + b0

    if not (p1['X'] <= x_intersection <= p2['X']) or np.isnan(x_intersection):
        return [-1, -1], -1, False
    print(f'valid intersection found!\n'
          f'at ({x_intersection}, {y_intersection})')
    # get q value of vector
    q = (x_intersection - X[0]) / n[0]
    return np.array([x_intersection, y_intersection]), q, True


threshold = 0.0001


# get point on next curve
def get_next_point(p, points, dt):
    inter_index = -1
    min_q = 0
    inter_coords = []
    # get current point position, and account for fluid flow
    X = np.array([p['X'] + p['x_velocity'] * dt, p['Y'] + p['y_velocity'] * dt])
    # get normal vector for direction
    n = np.array([p['progress_variable_gx'], p['progress_variable_gy']])
    # get y intercept of point line
    b0 = X[1] - (n[1]/n[0]) * X[0]
    # for every point
    for i in range(len(points) - 1):
        cur_p1 = points.iloc[i]
        cur_p2 = points.iloc[i+1]
        inter, q, success = find_intersection(X, n, b0, cur_p1, cur_p2)
        if not success:
            continue
        if inter_index == -1 or min_q > q:
            inter_index = i
            min_q = q
            inter_coords = inter

    if inter_index != -1:
        res = interpolate_point(inter_coords, points.iloc[inter_index], points.iloc[inter_index + 1], min_q, dt)
        print(f'inter coords: {inter_coords}, curvature: {res["curvature"]}')
    else:
        res = {'success': False}
    return res


def process_surfaces(index, smooth=False):
    # process all files
    file_names = []
    print('Formatting all files ...')
    for i in range(1000, 2001):
        # print(f'Formatting file {i} ...')
        if smooth:
            file_name = f'res/2d_paths/pathIso{str(i).zfill(5)}Reg_smooth'
        else:
            file_name = f'res/2d_paths/pathIso{str(i).zfill(5)}Reg'
        # format_all_dat(file_name)
        file_names.append(file_name)
    print('All files formatted:')
    isoT0 = pd.read_pickle(f'{file_names[0]}.pkl')
    p0 = isoT0.iloc[index]

    all_points = []
    last_p0 = p0
    for i in range(1001, 2001):
        print(i)
        isoT1 = pd.read_pickle(f'{file_names[i-1000]}.pkl')
        # print(isoT1)
        isoT1['dist'] = isoT1.apply(lambda row: VectorOps.point_dist(row, p0), axis=1)
        isoT1 = isoT1[isoT1['dist'] < threshold]
        t0 = p0['t']
        t1 = isoT1.iloc[0]['t']
        dt = t1 - t0
        p0 = get_next_point(p0, isoT1, dt)
        if not p0['success']:
            print('Failed!')
            return
        p0['dr'] = p0['r'] - last_p0['r']
        all_points.append(p0)
        last_p0 = p0

    path_data = pd.DataFrame.from_dict(all_points)
    path_data.dropna()
    if smooth:
        path_data.to_pickle(f'res/2d_paths/{index}PathData_smooth.pkl')
    else:
        path_data.to_pickle(f'res/2d_paths/{index}PathData.pkl')
    plt.show()

    return p0


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


def update_fig(t, data, ax_an1, ax_an2):
    step = int(t) % 800
    print(t)
    # print(data)
    cur_point = data.iloc[step]
    global surfaces
    cur_surface = surfaces[step]
    ax_an1.cla()
    ax_an1.set_xlim(0, 0.032)
    ax_an1.set_ylim(0, 0.032)
    ax_an1.set_xlabel('X')
    ax_an1.set_ylabel('Y')
    n = np.array([cur_point['progress_variable_gx'] * cur_point['q'], cur_point['progress_variable_gy'] * cur_point['q']])
    u = np.array([cur_point['x_velocity'] * cur_point['dt'], cur_point['y_velocity'] * cur_point['dt']])
    dif = u + n
    ax_an1.plot(cur_surface['X'], cur_surface['Y'])
    ax_an1.scatter(cur_point['X'], cur_point['Y'])
    ax_an1.quiver(cur_point['X'], cur_point['Y'], n[0], n[1], angles='xy', scale_units='xy', scale=1)
    ax_an1.quiver(cur_point['X'], cur_point['Y'], u[0], u[1], angles='xy', scale_units='xy', scale=1)
    ax_an1.quiver(cur_point['X'], cur_point['Y'], dif[0], dif[1], angles='xy', scale_units='xy', scale=1)
    ax_an2.cla()
    ax_an2.plot(data['t'], data['k_hat'])
    ax_an2.set_xlabel('t')
    ax_an2.set_ylabel('curvature')
    ax_an2.scatter(cur_point['t'], cur_point['curvature'])
    return ax_an1, ax_an2


surfaces = []


def graph_data(index, smooth=False, animating=False):
    print(f'reading data...')
    global surfaces
    if smooth:
        data = pd.read_pickle(f'res/2d_paths/{index}PathData_smooth.pkl')
        if animating:
            for i in range(1000, 2001):
                surfaces.append(pd.read_pickle(f'res/2d_paths/pathIso0{i}Reg_smooth.pkl'))
    else:
        data = pd.read_pickle(f'res/2d_paths/{index}PathData.pkl')
    data = data[200:]

    data['k_hat'] = savgol_filter(data['curvature'], 50, 3)
    data['dr_hat'] = (1/data['k_hat'] - 1/data['k_hat'].shift(1))
    # data['dr_hat'] = savgol_filter(data['dr'], 50, 3)
    data['drdt'] = data['dr_hat'] / data['dt']
    data['kdrdt'] = data['k_hat'] * data['dr_hat'] / data['dt']
    # data['kdrdt'] = savgol_filter(data['kdrdt'], 50, 3)
    data = data.dropna()

    if animating:
        surfaces = surfaces[200:]
        print(f'animating ...')
        fig = plt.figure()
        ax_an1 = fig.add_subplot(1, 2, 1)
        ax_an2 = fig.add_subplot(1, 2, 2)
        anim = animation.FuncAnimation(fig, update_fig, fargs=(data, ax_an1, ax_an2),
                                       interval=1, frames=800, blit=False)
        fig.tight_layout()
        plt.show()
        print(f'saving ...')
        # writervideo = animation.FFMpegWriter(fps=60)
        # anim.save('test_anim.mp4', writer=writervideo)
        plt.close()
        print(f'saved!')

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


def main():
    # new_curvature()
    # process_dats(smooth=True)0
    # regular_process(smooth=True)
    # process_surfaces(50000, smooth=True)
    graph_data(50000, smooth=True, animating=True)
    # test_stuff()
    return


if __name__ == '__main__':
    main()
