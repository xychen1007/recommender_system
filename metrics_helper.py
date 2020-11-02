import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def MAE(test, pred):
    return np.nanmean(np.absolute(test-pred))


def MSE(test, pred):
    n = test.shape[0]*test.shape[1] - np.isnan(pred)
    return np.nansum((test-pred)**2)/n


def dcg(test, pred):
    m = test.shape[1]
    sorted_pred = -np.sort(-pred, axis=0)
    sorted_indice = np.argsort(-pred, axis=0)
    sorted_test = test[sorted_indice]
    ranking = np.zeros(test.shape)
    for i in range(m):
        ranking[:][i] = i + 1
    return np.mean(np.nansum((np.where(np.isnan(sorted_test), np.nan, 1))/np.log2(ranking+1), axis=1))
