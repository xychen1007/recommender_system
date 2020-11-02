import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def pearson_similarity(vec1, vec2):
    mean_vec1 = np.nanmean(vec1)
    mean_vec2 = np.nanmean(vec2)
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for i in range(len(vec1)):
        if (not np.isnan(vec1[i])) and (not np.isnan(vec2[i])):
            numerator += (vec1[i]-mean_vec1)*(vec2[i]-mean_vec2)
            denominator1 += (vec1[i]-mean_vec1)**2
            denominator2 += (vec2[i]-mean_vec2)**2
    return numerator/np.sqrt(denominator1*denominator2)


def consine_similarity(vec1, vec2):
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for i in range(len(vec1)):
        if (not np.isnan(vec1[i])) and (not np.isnan(vec2[i])):
            numerator += vec1[i]*vec2[i]
            denominator1 += vec1[i]**2
            denominator2 += vec2[i]**2
    return numerator/np.sqrt(denominator1*denominator2)


def sim_matrix(data, metric):
    sim_matrix = np.zeros((len(data), len(data)))
    if metric == 'pearson':
        for i in range(len(sim_matrix)):
            sim_matrix[i][i] = 1
            for j in range(i+1, len(sim_matrix[i])):
                sim_matrix[i][j] = pearson_similarity(data[i], data[j])
                sim_matrix[j][i] = sim_matrix[i][j]
    elif metric == 'cosine':
        for i in range(len(data)):
            sim_matrix[i][i] = 1
            for j in range(i+1, len(data[i])):
                sim_matrix[i][j] = pearson_similarity(data[i], data[j])
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix


class ItemBasedCF():
    def __init__(self, k_neighbors, similarity_metrics):
        self.train_set = None
        self.similarity_matrix = None
        self.k_neighbors = k_neighbors
        self.similarity_metrics = similarity_metrics

    def train(self, train_df):
        self.train_set = train_df.values
        self.similarity_matrix = sim_matrix(self.train_set, self.similarity_metrics)

    def predict(self, test_df):
        pred_df = np.zeros(test_df.shape)
        train_set_filled = np.nan_to_num(self.train_set)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors+1, algorithm='ball_tree').fit(train_set_filled)
        distances, indices = nbrs.kneighbors(train_set_filled)
        mean_item = np.nanmean(self.train_set, axis=1)
        for i in range(len(pred_df)):
            pred_df[i][:] = mean_item[i]
            neighbor_indice = indices[i][1:]
            for j in range(len(pred_df[i])):
                pred_df[i][j] += np.nansum(self.similarity_matrix[i][neighbor_indice] * (
                            self.train_set[:, j][neighbor_indice] - mean_item[neighbor_indice])) / np.sum(
                    self.similarity_matrix[i][neighbor_indice])
        return pred_df

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./ml-latest/ratings.csv')
df.drop("timestamp", inplace=True, axis=1)

def sample_df(df, min_item=300, min_user=6800):
    countItem = df[['movieId', 'rating']].groupby(['movieId']).count()
    countUser = df[['userId', 'rating']].groupby(['userId']).count()

    selectedItemId = countItem.loc[countItem['rating'] > min_user].index
    selectedUserId = countUser.loc[countUser['rating'] > min_item].index

    n_users = len(selectedUserId)
    n_items = len(selectedItemId)
    print(f'number of users: {n_users}')
    print(f'number of items: {n_items}')
    df_sample = df.loc[(df['movieId'].isin(selectedItemId))&(df['userId']).isin(selectedUserId)]
    print(f'shape of sampled df: {df_sample.shape}')
    return df_sample

df_sample = sample_df(df)
train_set, test_set = train_test_split(df_sample, test_size=0.50)
train_set, valid_set = train_test_split(train_set, test_size=0.2)
rating_matrix_train = train_set.pivot(index='movieId', columns='userId', values='rating')
rating_matrix_test = test_set.pivot(index='movieId', columns='userId', values='rating')

train_set, test_set = train_test_split(df_sample, test_size=0.50)
train_set, valid_set = train_test_split(train_set, test_size=0.2)

rating_matrix_train = train_set.pivot(index='movieId', columns='userId', values='rating')
rating_matrix_test = test_set.pivot(index='movieId', columns='userId', values='rating')

model1 = ItemBasedCF(10, 'pearson')

import time
start = time.time()
model1.train(rating_matrix_train)
end = time.time()
print(f'training_time{end-start}')

start = time.time()
pred_set = model1.predict(rating_matrix_test)
end = time.time()
print(f'predicting_time{end-start}')