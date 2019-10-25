import sys
import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, BaselineOnly
from surprise import SVD, SVDpp, NMF
from surprise.reader import Reader
from surprise.model_selection.split import train_test_split
from surprise import accuracy

if __name__ == "__main__":
    df = pd.read_csv('../data/movies/ratings.csv')
    df.drop('timestamp', axis=1, inplace=True)
    reader = Reader()
    data = Dataset.load_from_df(df, reader=reader)
    print('Using ALS')
    bsl_options = {'method': 'als',
                'n_epochs': 5,
                'reg_u': 12,
                'reg_i': 5
                }

    trainset, testset = train_test_split(data, test_size=0.25)
    algo = BaselineOnly(bsl_options=bsl_options)
    predictions = algo.fit(trainset).test(testset)
    accuracy.rmse(predictions)