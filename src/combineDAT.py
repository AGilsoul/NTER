import numpy as np
import pandas as pd
import formatDAT as fd

prog_var = 'H2O'
c_range = np.arange(0.05, 1, 0.05)
file_prefix = f'res/isoSurfaces_C_{prog_var}/plt01929_iso_Y({prog_var})_'


dataframes = []
fd.formatDAT(file_prefix)

for c in c_range:
    value = float("{:.2f}".format(c))
    file_name = rf'{file_prefix}{"{:.2f}".format(c)}.dat'
    print(f'Opening {file_name} ...')
    data = pd.read_csv(file_name, delimiter='\s+', dtype=float).dropna()
    C_list = [value for _ in range(len(data))]
    # C_H2_list = [1 - (row['Y_H2']/0.011608) for index, row in data.iterrows()]
    # C_H2O_list = [row['Y_H2O']/0.10371 for index, row in data.iterrows()]
    # data['C_H2'] = C_list
    data[f'C_{prog_var}'] = C_list
    dataframes.append(data)

final_data = pd.concat(dataframes)
pd.to_pickle(final_data, f'res/plt01929_iso_CRange_Y{prog_var}.pkl')
