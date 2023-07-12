import cantera as ct

gas = ct.Solution('res/mechanism.yaml')
gas.set_equivalence_ratio(phi=0.4, fuel="H2:1.0", oxidizer="O2:1.0,N2:3.76")

gas.equilibrate('HP')
gas()

rf = gas.forward_rates_of_progress
rr = gas.reverse_rates_of_progress
for i in range(gas.n_reactions):
    if gas.is_reversible(i) and rf[i] != 0.0:
        print(' %4i  %10.4g  ' % (i, (rf[i] - rr[i])/rf[i]))
