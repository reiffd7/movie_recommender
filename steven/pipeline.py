import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd

from surprise import AlgoBase, Dataset, BaselineOnly
from surprise import SVD, SVDpp, NMF
from surprise.reader import Reader
from surprise.model_selection.split import train_test_split
from surprise import accuracy
from steven.ratings_residuals_histogram import single_histogram, double_histogram
from surprise.model_selection.validation import cross_validate

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
DATA_DIRECTORY = os.path.join(os.path.split(FILE_DIRECTORY)[0], 'data', 'movies')

if __name__ == "__main__":
    # Read data
    df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'ratings.csv'))

    # Drop unneeded column 'timestamp'
    df.drop('timestamp', axis=1, inplace=True)

    # Load the data into the surprise format
    reader = Reader()
    data = Dataset.load_from_df(df, reader=reader)

    # Train ALS model
    print('Using ALS')
    bsl_options = {'method': 'als',
                'n_epochs': 5,
                'reg_u': 12,
                'reg_i': 5
                }
    trainset, testset = train_test_split(data, test_size=0.25)
    algo = BaselineOnly(bsl_options=bsl_options)
    predictions = algo.fit(trainset).test(testset)

    # Get the RMSE of our predictions
    rmse = accuracy.rmse(predictions)

    # Get the cross-validated RMSE of our predictions
    cv_results = cross_validate(algo, data)
    cv_rmse = cv_results['test_rmse'].mean()
    print(f'CV RMSE: {cv_rmse}')

    # Get true values and predicted values for our test set
    y_true = [x.r_ui for x in predictions]
    y_pred = [x.est for x in predictions]

    # Plot a histogram of our model residuals
    single_histogram(y_true, y_pred, title=f'Histogram of ALS Model Residuals - RMSE {cv_rmse:.3f}')
