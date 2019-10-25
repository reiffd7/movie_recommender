#!/usr/bin/env python

"""
http://surprise.readthedocs.io/en/stable/building_custom_algo.html
"""

import sys
import numpy as np
from surprise import AlgoBase, Dataset, BaselineOnly, Reader, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
import pandas as pd


class GlobalMean(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean


class MeanofMeans(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        users = np.array([u for (u, _, _) in self.trainset.all_ratings()])
        items = np.array([i for (_, i, _) in self.trainset.all_ratings()])
        ratings = np.array([r for (_, _, r) in self.trainset.all_ratings()])

        user_means,item_means = {},{}
        for user in np.unique(users):
            user_means[user] = ratings[users==user].mean()
        for item in np.unique(items):
            item_means[item] = ratings[items==item].mean()

        self.global_mean = ratings.mean()    
        self.user_means = user_means
        self.item_means = item_means
                            
    def estimate(self, u, i):
        """
        return the mean of means estimate
        """
        
        if u not in self.user_means:
            return(np.mean([self.global_mean,
                            self.item_means[i]]))

        if i not in self.item_means:
            return(np.mean([self.global_mean,
                            self.user_means[u]]))

        return(np.mean([self.global_mean,
                        self.user_means[u],
                        self.item_means[i]]))




if __name__ == "__main__":

    # data = Dataset.load_builtin('ml-100k')
    # print("\nGlobal Mean...")
    # algo = GlobalMean()
    # evaluate(algo, data)

    # print("\nMeanOfMeans...")
    # algo = MeanofMeans()
    # evaluate(algo, data)

    # bsl_options = {'method': 'als',
    #            'n_epochs': 10,
    #            'reg_u': 2,
    #            'reg_i': 5
    #            }
    # algo = BaselineOnly(bsl_options=bsl_options)
    # evaluate(algo, data)
    ratings_df = pd.read_csv('../data/movies/ratings.csv')
    reader = Reader(name=None, line_format=u'user item rating', sep=',', rating_scale=(1, 5), skip_lines=0)
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Perform cross validation
        print(algorithm)
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        print(results)
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
        print("appended to benchmark")
        
    benchmark_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
    
