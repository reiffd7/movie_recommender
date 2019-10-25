import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances

movies = pd.read_csv('../data/movies/movies.csv')

def one_hot_encode_genres(df):
    '''
    splits out movies genre columns to one-hot encode by genre
    '''
    df.genres = df.genres.apply(lambda row: row.split('|'))

    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres')),
                            columns=mlb.classes_,
                            index=df.index))
    df.rename(columns={'(no genres listed)':'none'}, inplace=True)
    return df

def similarities(df):
    '''
    returns cosine similarity matrix by index and by title
    '''
    
    dist_out = 1-pairwise_distances(movies.iloc[:,2:], metric="cosine")
    simimlarities = pd.DataFrame(dist_out)

    title_sims = simimlarities.copy()
    title_sims.index = movies.title.replace(' ','_')
    title_sims.columns = movies.title.replace(' ','_')
    
    return simimlarities, title_sims

def similar_by_genre(df, movie, num):
    '''
    prints list of num most similar movies by genre
    input: df = similarity dataframe, movie = movie title (or index), 
    num = number of movies to generate
    '''
    similar_list = list(np.argsort(df[movie])[::-1])[1:num+1]
    similar_movies = []
    for i in similar_list:
        similar_movies.append(df.iloc[0].index[i])
    print(similar_movies)
    return similar_movies

if __name__ == "__main__":
    movies = pd.read_csv('../data/movies/movies.csv')
    movies = one_hot_encode_genres(movies)
    simimlarities, title_sims = similarities(movies)
    similar_by_genre(title_sims,'GoldenEye (1995)',10)
    