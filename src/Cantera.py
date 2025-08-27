import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def computeProgVar(Y_H2):
    Y_H2_u = 0.011608
    res = []
    for y in Y_H2:
        res.append(1 - (y/Y_H2_u))
    return res


def computeMixFrac(Y_H2, Y_O2):
    res = []
    for i in range(len(Y_H2)):
        res.append(0.011608)
    return res


def compute1DFlame():
    gas = ct.Solution('res/LiDryer.yaml')
    gas.set_equivalence_ratio(phi=0.4, fuel="H2:1.0", oxidizer="O2:1.0,N2:3.76")
    gas.TP = 300, ct.one_atm
    # gas.equilibrate('HP')
    # gas()
    grid = np.linspace(0, 0.0012, 999)
    flame = ct.FreeFlame(gas, grid=grid)
    flame.transport_model = 'Multi'
    flame.solve(loglevel=0, auto=True)
    Y_H2 = flame.Y[0]
    Y_O2 = flame.Y[1]
    print(flame.velocity[0])
    return list(flame.T), list(computeProgVar(Y_H2)), list(computeMixFrac(Y_H2, Y_O2)), list(grid)


def compute1DStrain():
    def derivative(x, y):
        dydx = np.zeros(y.shape, y.dtype.type)

        dx = np.diff(x)
        dy = np.diff(y)
        dydx[0:-1] = dy / dx

        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dydx

    def computeStrainRates(oppFlame):
        strainRates = derivative(oppFlame.grid, oppFlame.velocity)

        maxStrLocation = abs(strainRates).argmax()
        minVelocityPoint = oppFlame.velocity[:maxStrLocation].argmin()

        strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
        K = abs(strainRates[strainRatePoint])

        return strainRates, strainRatePoint, K

    def computeConsumptionSpeed(oppFlame):
        Tb = max(oppFlame.T)
        Tu = min(oppFlame.T)
        rho_u = max(oppFlame.density)

        integrand = oppFlame.heat_release_rate / oppFlame.cp

        total_heat_release = np.trapz(integrand, oppFlame.grid)
        Sc = total_heat_release / (Tb - Tu) / rho_u

        return Sc

    def solveOpposedFlame(oppFlame, massFlux=0.12, loglevel=1, ratio=10, slope=0.1, curve=0.1):
        oppFlame.reactants.mdot = massFlux
        oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve)
        oppFlame.set_time_step(1.0e-5, [2, 5, 10, 20, 50])
        # oppFlame.show_solution()
        oppFlame.solve(loglevel, refine_grid=False, auto=True)

        strainRates, strainRatePoint, K = computeStrainRates(oppFlame)

        return np.max(oppFlame.T), K, strainRatePoint

    av_range = np.linspace(1.0, 2.0, num=100)
    K_range = []
    T_range = []
    Sc_range = []
    tol_ss = [1.0e-11, 1.0e-13]
    tol_ts = [1.0e-11, 1.0e-13]

    initial_grid = np.linspace(0, 0.032, 1000)
    print(initial_grid.shape)

    for i in av_range:
        gas = ct.Solution('res/LiDryer.yaml')
        gas.set_equivalence_ratio(phi=0.4, fuel="H2:1.0", oxidizer="O2:1.0,N2:3.76")
        gas.TP = 300, ct.one_atm

        oppFlame = ct.CounterflowTwinPremixedFlame(gas, grid=initial_grid)
        oppFlame.transport_model = 'Multi'

        # oppFlame.flame.set_steady_tolerances(default=tol_ss)
        # oppFlame.flame.set_transient_tolerances(default=tol_ts)

        print(f'Cur axial velocity: {i} m/s')
        massFlux = gas.density * i
        (T, K, strainRatePoint) = solveOpposedFlame(oppFlame, massFlux, loglevel=0)
        Sc = computeConsumptionSpeed(oppFlame)
        K_range.append(K)
        T_range.append(T)
        Sc_range.append(Sc)

    K_range = np.array(K_range)
    T_range = np.array(T_range)
    Sc_range = np.array(Sc_range)

    data_array = np.array([av_range, K_range, T_range, Sc_range]).T

    df = pd.DataFrame(data_array,
                      columns=['Axial Velocity (m/s)', 'Strain K', 'Temperature T (K)', 'Consumption Speed'])
    df.to_csv('strain_data')

    print(f'K range: {K_range}')
    print(f'T range: {T_range}')
    print(f'Sc range: {Sc_range}')
    plt.scatter(K_range, T_range)
    plt.ylabel('Temperature T (K)')
    plt.xlabel('Strain K')
    plt.show()


def main():
    # compute1DStrain()
    compute1DFlame()


if __name__ == '__main__':
    main()

