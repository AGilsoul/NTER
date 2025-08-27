import re
from itertools import islice
import pandas as pd


def ReadDat(file_name):
    isoTime = 0
    with open(file_name) as fin:
        for line in islice(fin, 1, 2):
            isoTime = float(re.findall('"([^"]*)"', line)[0])
            break
    data = pd.read_csv(file_name, skiprows=[1], delimiter='\s+', dtype=float).dropna()
    return isoTime, data