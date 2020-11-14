import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from metrics_helper_v3 import *


def stratify_sample(df, n_item):
    ratingsByMovie = df[['movieId', 'rating']].groupby(['movieId']).count()
    movie1 = random.sample(list(ratingsByMovie[ratingsByMovie.values<2].index), n_item)
    movie2 = random.sample(list(ratingsByMovie[(ratingsByMovie.values>=2)&(ratingsByMovie.values<4)].index), n_item)
    movie3 = random.sample(list(ratingsByMovie[(ratingsByMovie.values>=4)&(ratingsByMovie.values<10)].index), n_item)
    movie4 = random.sample(list(ratingsByMovie[(ratingsByMovie.values>=10)&(ratingsByMovie.values<50)].index), n_item)
    movie5 = random.sample(list(ratingsByMovie[ratingsByMovie.values>=50].index), n_item)
    df1 = df.loc[df['movieId'].isin(movie1+movie2+movie3+movie4), :]
    df5 = df.loc[df['movieId'].isin(movie5), :]
    selected_user = random.sample(df5['userId'].unique().tolist(), 20000)
    df5 = df5.loc[df['userId'].isin(selected_user), :]
    return df1.append(df5)


def sample_df(df, user_thresh=300, item_thresh=6800):
    countItem = df[['movieId', 'rating']].groupby(['movieId']).count()
    countUser = df[['userId', 'rating']].groupby(['userId']).count()

    selectedItemId = countItem.loc[countItem['rating'] > item_thresh].index
    selectedUserId = countUser.loc[countUser['rating'] > user_thresh].index

    n_users = len(selectedUserId)
    n_items = len(selectedItemId)
    print(f'number of users: {n_users}')
    print(f'number of items: {n_items}')
    df_sample = df.loc[(df['movieId'].isin(selectedItemId)) & (df['userId']).isin(selectedUserId)]
    print(f'shape of sampled df: {df_sample.shape}')
    return df_sample

def plot_grid_search(cv_results_metric, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = np.array(cv_results_metric).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    
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


def cosine_similarity(vec1, vec2):
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
        for i in range(len(sim_matrix)):
            sim_matrix[i][i] = 1
            for j in range(i+1, len(sim_matrix[i])):
                sim_matrix[i][j] = cosine_similarity(data[i], data[j])
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix



class ItemBasedCF():
    def __init__(self, k_neighbors=10, similarity_metrics='pearson'):
        self.train_set = None
        self.similarity_matrix = None
        self.k_neighbors = k_neighbors
        self.similarity_metrics = similarity_metrics

    def fit(self, train_df, dummy_y):
        self.train_df = train_df.pivot(index='movieId', columns='userId', values='rating')
        self.train_set = self.train_df.values
        self.similarity_matrix = sim_matrix(self.train_set, self.similarity_metrics)

    def predict(self):
        pred_df = np.zeros(self.train_df.shape)
        train_set_filled = np.nan_to_num(self.train_set)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree').fit(train_set_filled)
        distances, indices = nbrs.kneighbors(train_set_filled)

        # return average score of each movie with NAN ignored
        mean_item = np.nanmean(self.train_set, axis=1)
        for i in range(len(pred_df)):
            pred_df[i][:] = mean_item[i]
            neighbor_indice = indices[i][1:]
            for j in range(len(pred_df[i])):
                if np.isnan(self.train_set[i][j]):
                    pred_df[i][j] += np.nansum(self.similarity_matrix[i][neighbor_indice] * (
                            self.train_set[:, j][neighbor_indice] - mean_item[neighbor_indice])) / np.sum(
                        self.similarity_matrix[i][neighbor_indice])
                else:
                    pred_df[i][j] = np.nan
        return pred_df
    
    def score(self, X, y):
        pred_df = self.predict()
        test_df = self.train_df.copy()
        test_df[:][:] = np.nan
        y = y.pivot(index='movieId', columns='userId', values='rating')
        for i in test_df.index:
            for j in test_df.columns:
                try:
                    test_df.loc[i][j] = y.loc[i][j]
                except:
                    pass
        return MAE_cv(test_df, pred_df)

    # requested methods for sk-learn estimator
    def get_params(self, deep=True):
        return {"k_neighbors": self.k_neighbors, "similarity_metrics": self.similarity_metrics}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# if __name__ == "__main__":
#     import numpy as np
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#
#     df = pd.read_csv('./ml-latest/ratings.csv')
#     df.drop("timestamp", inplace=True, axis=1)
#
#     df_sample = sample_df(df)
#     train_set, test_set = train_test_split(df_sample, test_size=0.50)
#     train_set, valid_set = train_test_split(train_set, test_size=0.2)
#     rating_matrix_train = train_set.pivot(index='movieId', columns='userId', values='rating')
#     rating_matrix_test = test_set.pivot(index='movieId', columns='userId', values='rating')
#
#     model1 = ItemBasedCF(10, 'pearson')
#
#     import time
#     start = time.time()
#     model1.train(rating_matrix_train)
#     end = time.time()
#     print(f'training_time{end-start}')
#
#     start = time.time()
#     pred_set = model1.predict(rating_matrix_test)
#     end = time.time()
#     print(f'predicting_time{end-start}')