import pandas as pd
import numpy as np


def CSVtoPKL(file_prefix):
    print('reading csv...')
    data = pd.read_csv(f'res/{file_prefix}.csv', dtype=np.float32).dropna()
    print('saving to pkl...')
    pd.to_pickle(data, f'res/{file_prefix}.pkl')
    print('done!')


