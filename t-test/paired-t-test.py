import numpy as np
import pandas as pd
import math


def paired_t_test():
    X = pd.read_excel(open('metric-5_3_1.xlsx', 'rb'), sheet_name='exp-1', index_col=0).to_numpy()
    Y = pd.read_excel(open('metric-4_3_1.xlsx', 'rb'), sheet_name='exp-1', index_col=0).to_numpy()
    # diff = np.zeros((14, 20))
    print(X.shape)
    print(Y.shape)
    t_list = []
    diff_len = []
    for row in range(0, min(X.shape[0], Y.shape[0])):
        X_list = []
        Y_list = []
        for col in range(0, 30):
            if X[row][col] > 0:
                X_list.append(X[row][col])
            if Y[row][col] > 0:
                Y_list.append(Y[row][col])
        if len(X_list) < len(Y_list):
            Y_list = Y_list[:len(X_list)]
        elif len(X_list) > len(Y_list):
            X_list = X_list[:len(Y_list)]
        diff = np.absolute(np.array(X_list) - np.array(Y_list))
        mean = np.mean(diff)
        std = np.std(diff)
        std = std / math.sqrt(len(diff))
        t = np.divide(mean, std)
        print(t)
        # print(len(diff))



if __name__ == '__main__':
    paired_t_test()
