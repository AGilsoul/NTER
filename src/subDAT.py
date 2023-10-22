import pandas as pd
from ReadDat import ReadDat
from formatDAT import formatDAT

# should be placed in PeleAnalysis/Src/TemporalPath/Python/src

xlim = [0, 0.016]
ylim = [0, 0.016]

time_list = range(1370, 2050, 10)

for i in time_list:
    file = f'../../../path_out/pathIso0{i}.dat'
    formatDAT(file)
    time, df = ReadDat(file)
    df = df[(df['X'] < xlim[1]) & (df['X'] > xlim[0]) & (df['Y'] < ylim[1]) & (df['Y'] > ylim[0])]
    df['t'] = [time for _ in range(len(df))]
    pd.to_pickle(df, f'../../../path_out/subPathIso0{i}.pkl')


