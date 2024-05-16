# app.py

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from data_utils import train_texts

app = Flask(__name__)
app.debug = True  # Enable debug mode

# Load the trained sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

# Load the tokenizer used for preprocessing
tokenizer = Tokenizer(num_words=8000)  # Assuming the same tokenizer is used as during training
tokenizer.fit_on_texts(train_texts)

def predict_sentiment(review):
    # Preprocess the input review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=500)

    # Predict sentiment using the loaded model
    prediction = model.predict(padded_sequence)[0][0]
    if prediction >= 0.5:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('result.html', sentiment=sentiment, review=review)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run()
