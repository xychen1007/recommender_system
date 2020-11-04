import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def MAE(test, pred):
    return np.nanmean(np.absolute(test-pred))


def MSE(test, pred):
    return np.nanmean((test-pred)**2)


def RMSE(test, pred):
    return np.nanmean((test-pred)**2)**0.5


def dcg(test, pred):
    m = test.shape[1]
    sorted_pred = -np.sort(-pred, axis=0)
    sorted_indice = np.argsort(-pred, axis=0)
    sorted_test = np.zeros(test.shape)
    ranking = np.zeros(test.shape)
    for i in range(m):
        sorted_test[:, i] = test[:, i][sorted_indice[:, i]]
        ranking[:, i] = i + 1
    return np.mean(np.nansum((np.where(np.isnan(sorted_test), np.nan, 1))/np.log2(ranking+1), axis=1))


# Assume our strategy is recommending top k movies for each user
def reco2user(matrix, k_items=10):
    reco = np.zeros(matrix.shape)
    m = matrix.shape[1]
    topK = np.argsort(-matrix, axis=0)[:k_items,:]
    for i in range(m):
        indice = topK[:,i]     #the k movies that are recommended to user i
        reco[:,i][indice] = 1  #set the recommended movies as 1 
    return reco


def cal_coverage(test, pred, threshold=3, coverage_type='user'):
    # get recommendation matrix based on both actual and predictions
    reco_actual = reco2user(test, k_items=10)
    reco_pred =  reco2user(pred, k_items=10)
    # count the right coverage
    coverage = (reco_pred == reco_actual) & (reco_pred == 1) & (reco_actual == 1)
    if coverage_type == 'user':
        right_reco = np.sum(coverage, axis=0)
        return np.sum(np.where(right_reco>=threshold,1,0))/len(right_reco)
    elif coverage_type == 'item':
        right_reco = np.sum(coverage, axis=1)
        return np.sum(np.where(right_reco>=threshold,1,0))/len(right_reco)
    elif coverage_type == 'catalog':
        right_reco = np.sum(coverage, axis=1)
        return np.sum(np.where(right_reco>=1,1,0))/len(right_reco)
