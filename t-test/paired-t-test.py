import numpy as np
import pandas as pd
import math

def paired_t_test():
    X = pd.read_excel(open('metric-exp-1.xlsx', 'rb'), sheet_name='1', index_col=0).to_numpy()
    X = X[:, :20]
    Y = pd.read_excel(open('metric-exp-2.xlsx', 'rb'), sheet_name='exp-2', index_col=0).to_numpy()
    diff = np.zeros((14, 20))
    # print(diff.shape)
    for i in range(0, 14):
        for j in range(0, 20):
            if X[i][j] > 0 and Y[i][j] > 0:
                diff[i][j] = X[i][j] - Y[i][j]
            else:
                diff[i][j] = np.nan

    mean = np.nanmean(diff, axis=1)
    std = np.nanstd(diff, axis=1)
    print(np.nanmean(diff, axis=1))
    print(np.nanstd(diff, axis=1))
    t = np.divide(mean, std)
    # print(t)
    print(t/math.sqrt(20))



if __name__ == '__main__':
    paired_t_test()
