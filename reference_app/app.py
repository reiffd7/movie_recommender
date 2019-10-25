import pickle
import pandas as pd

from flask import Flask, render_template, request, jsonify

from build_model import TextClassifier, get_data

app = Flask(__name__)

# Load model
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

#################################
## TEST AJAX
#################################
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/test-ajax')
def test_ajax():
    return render_template('test-ajax.html')

#################################

##########################################
# Separate page-load response
##########################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = model.predict([request.form['user_input']])[0]
    #return f"Your predicted category for this text is: {prediction}"
    return render_template('predict.html', prediction=prediction)

##########################################
# AJAX submission and response
##########################################

@app.route('/submit-ajax')
def submit_ajax():
    return render_template('submit-ajax.html')

@app.route('/_predict')
def _predict():
    return jsonify(result=model.predict([request.args.get('user_input', '', type=str)])[0])




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
