import re
from itertools import islice
import pandas as pd


def ReadCsv(file_name):
    isoTime = 0
    data = pd.read_csv(file_name, dtype=float).dropna()
    print(data)
    return data