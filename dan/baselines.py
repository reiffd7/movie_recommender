#!/usr/bin/env python

"""
http://surprise.readthedocs.io/en/stable/building_custom_algo.html
"""

import sys
import numpy as np
from surprise import AlgoBase, Dataset, BaselineOnly, Reader, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd
import get_movie_name as steven


class GlobalMean(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean


class MeanOfMeans(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

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

        return self
                            
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





class cvWrapper(object):

    def __init__(self, ratings_df):
        self.ratings_df = ratings_df

    def load_data(self):
        reader = Reader(name=None, line_format=u'user item rating', sep=',', rating_scale=(1, 5), skip_lines=0)
        self.data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)

    def cv(self, algorithm_list):
        benchmark = []
        # Iterate over all algorithms
        for algorithm in algorithm_list:
            # Perform cross validation
            print(algorithm)
            results = cross_validate(algorithm, self.data, measures=['RMSE'], cv=3, verbose=False)
            print(results)
            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)
            print("appended to benchmark")
            
        self.benchmark_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')

    def split_train_predict(self, algo):
        algo = algo
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)
        self.predictions = algo.fit(self.trainset).test(self.testset)
        



def get_Iu(uid, trainset):
        """ return the number of items rated by given user
        args: 
        uid: the id of the user
        returns: 
        the number of items rated by the user
        """
        try:
            return len(trainset.ur[trainset.to_inner_uid(uid)])
        except ValueError: # user was not part of the trainset
            return 0


def get_Ui(iid, trainset):
    """ return number of users that have rated given item
    args:
    iid: the raw id of the item
    returns:
    the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0




    



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

    cv = cvWrapper(pd.read_csv('../data/movies/ratings.csv'))
    cv.load_data() 
    # algorithm_list = [SVD(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]
    # cv.cv(algorithm_list)

    bsl_options = {'method': 'als',
                    'n_epochs': 5,
                    'reg_u': 12,
                    'reg_i': 5
                    }
    algo = BaselineOnly(bsl_options=bsl_options)
    cv.split_train_predict(algo)

    predictions = cv.predictions
    trainset = cv.trainset
    testset = cv.testset
    print('Predictions made')
    df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    print('Adding items')
    df['Iu'] = df.uid.apply(get_Iu)
    print('Adding users')
    df['Ui'] = df.iid.apply(get_Ui)
    print('Adding error')
    df['err'] = abs(df.est - df.rui)
    df['iid'] = df.iid.apply(lambda x: steven.get_movie_names(x)[0])

    best_predictions = df.sort_values(by='err')[:100]
    worst_predictions = df.sort_values(by='err')[-100:]

    # ratings_df = pd.read_csv('../data/movies/ratings.csv')
    # reader = Reader(name=None, line_format=u'user item rating', sep=',', rating_scale=(1, 5), skip_lines=0)
    # data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    # benchmark = []
    # # Iterate over all algorithms
    # for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    #     # Perform cross validation
    #     print(algorithm)
    #     results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    #     print(results)
    #     # Get results & append algorithm name
    #     tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    #     tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    #     benchmark.append(tmp)
    #     print("appended to benchmark")
        
    # benchmark_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
    
