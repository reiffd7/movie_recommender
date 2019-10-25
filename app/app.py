import os
import pickle

from flask import Flask, render_template, jsonify, request, redirect, url_for

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
DATA_DIRECTORY = os.path.join(FILE_DIRECTORY, 'data')

app = Flask(__name__)

with open(os.path.join(DATA_DIRECTORY, 'user_recommendations.pkl'), 'rb') as f:
    user_recommendations = pickle.load(f)

with open(os.path.join(DATA_DIRECTORY, 'user_fave_movies.pkl'), 'rb') as f:
    user_favorites = pickle.load(f)

def get_movies(user_id):
    return user_recommendations[user_id]
    #return ['movie1', 'movie2', 'movie3']

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-favorite-movies')
def get_favorite_movies():
    return render_template('get-favorite-movies.html')

@app.route('/favorite-movies', methods=['GET', 'POST'])
def favorite_movies():
    user_id = request.form['user_id']
    if not user_id:
        return redirect('/')
    user_id = int(user_id)
    return render_template('favorite-movies.html', favorite_movies=user_favorites[user_id][:10])

@app.route('/movie-recommendations', methods=['GET', 'POST'])
def movie_recommendations():
    user_id = request.form['user_id']
    if not user_id:
        return redirect('/')
    user_id = int(user_id)

    num_movies = request.form['number_of_movies']
    if not num_movies:
        num_movies = 10
    num_movies = int(num_movies)

    movies = get_movies(user_id)
    movies = movies[:num_movies]

    return render_template('movie-recommendations.html', user_id=user_id, movies=movies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False) # Make sure to change debug=False for production
