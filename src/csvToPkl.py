import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from modin import config
from distributed import Client


def CSVtoPKL(file_prefix):
    client = Client()
    freeze_support()
    config.MinPartitionSize.put(128)
    print('reading csv...')
    data = pd.read_csv(f'res/{file_prefix}.csv', dtype=np.float32).dropna()
    print('saving to pkl...')
    pd.to_pickle(data, f'res/{file_prefix}.pkl')
    print('done!')


