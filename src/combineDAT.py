import numpy as np
import pandas as pd
import formatDAT as fd


def main():
    prog_var = 'H2O'
    c_range = [0.8, 1, 1.1]
    file_prefix = f'res/isoSurfaces_C_{prog_var}/plt01929_iso_Y({prog_var})_'

    dataframes = []
    fd.formatDAT(file_prefix, c_range)

    for c in c_range:
        value = float("{:.2f}".format(c))
        file_name = rf'{file_prefix}{"{:.2f}".format(c)}.dat'
        print(f'Opening {file_name} ...')
        data = pd.read_csv(file_name, delimiter='\s+', dtype=float).dropna()
        dataframes.append(data)

    final_data = pd.concat(dataframes)
    pd.to_pickle(final_data, f'res/plt01929_iso_CRange_Y{prog_var}.pkl')


if __name__ == '__main__':
    main()