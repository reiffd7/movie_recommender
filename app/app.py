from flask import Flask, render_template, jsonify, request
app = Flask(__name__)

def get_movies(user_id):
    return ['movie1', 'movie2', 'movie3']

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movie-recommendations', methods=['GET', 'POST'])
def movie_recommendations():
    user_id = int(request.form['user_id'])
    num_movies = int(request.form['number_of_movies'])

    movies = get_movies(user_id)
    movies = movies[:num_movies]

    return render_template('movie-recommendations.html', user_id=user_id, movies=movies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True) # Make sure to change debug=False for production
