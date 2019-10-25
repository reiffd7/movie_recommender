import os

import numpy as np
import pandas as pd

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
DATA_DIRECTORY = os.path.join(os.path.split(FILE_DIRECTORY)[0], 'data', 'movies')

MOVIES_DF = pd.read_csv(os.path.join(DATA_DIRECTORY, 'movies.csv'))

def get_movie_names(movie_ids):
    """Expects movie_ids as an int, list or NumPy array. Returns a NumPy array of strings.
    
    Example usage:
        >>> get_movie_names(1)
        array(['Toy Story (1995)'], dtype=object)

        >>> get_movie_names([1,2,3])
        array(['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)'],
            dtype=object)

        >>> get_movie_names(np.array([1,2,3]))
        array(['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)'],
            dtype=object)
    """
    if isinstance(movie_ids, int):
        return MOVIES_DF[MOVIES_DF['movieId'] == movie_ids]['title'].values
    if isinstance(movie_ids, list):
        movie_ids = np.array(movie_ids)
    return MOVIES_DF[MOVIES_DF['movieId'].isin(movie_ids)]['title'].values
