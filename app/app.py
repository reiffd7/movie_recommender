from flask import Flask, render_template, jsonify, request, redirect, url_for
app = Flask(__name__)

def get_movies(user_id):
    return ['movie1', 'movie2', 'movie3']

# home page
@app.route('/')
def index():
    return render_template('index.html')

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
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True) # Make sure to change debug=False for production
