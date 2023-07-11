import numpy as np
import pandas as pd

c_range = np.arange(0.05, 1, 0.05)

dataframes = []

for c in c_range:
    file_name = rf'res/plt01929_iso_{"{:.2f}".format(c)}.dat'
    print(f'Opening {file_name} ...')
    data = pd.read_csv(file_name, delimiter='\s+', dtype=float).dropna()
    c_list = [c for _ in range(len(data))]
    data['C_H2'] = c_list
    dataframes.append(data)

final_data = pd.concat(dataframes)
pd.to_pickle(final_data, 'res/plt01929_iso_CRange.pkl')
