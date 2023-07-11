import cantera

import cantera as ct

gas = ct.Solution('res/LiDryer/mechanism.yaml')
gas.set_equivalence_ratio(phi=0.4, fuel="H2:1.0", oxidizer="O2:1.0,N2:3.76")

gas()
