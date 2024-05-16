import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from data_utils import train_texts

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

def main():
    st.title('Sentiment Analysis App')

    # User input
    review = st.text_area('Enter your review here:', '')

    # Predict sentiment on button click
    if st.button('Analyze'):
        if review.strip() == '':
            st.error('Please enter a review.')
        else:
            sentiment = predict_sentiment(review)
            st.write('Sentiment:', sentiment)

if __name__ == '__main__':
    main()
