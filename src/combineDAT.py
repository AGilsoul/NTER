import numpy as np
import pandas as pd
import formatDAT as fd

c_range = np.arange(0.05, 1, 0.05)
file_prefix = '../res/isoSurfaces_C_H2/plt01929_iso_'

dataframes = []
fd.formatDAT(file_prefix)

for c in c_range:
    value = float("{:.2f}".format(c))
    file_name = rf'{file_prefix}{"{:.2f}".format(c)}.dat'
    print(f'Opening {file_name} ...')
    data = pd.read_csv(file_name, delimiter='\s+', dtype=float).dropna()
    C_H2_list = [value for _ in range(len(data))]
    # C_H2_list = [1 - (row['Y_H2']/0.011608) for index, row in data.iterrows()]
    C_H2O_list = [row['Y_H2O']/0.10371 for index, row in data.iterrows()]
    data['C_H2'] = C_H2_list
    data['C_H2O'] = C_H2O_list
    dataframes.append(data)

final_data = pd.concat(dataframes)
pd.to_pickle(final_data, '../res/plt01929_iso_CRange_YH2.pkl')
