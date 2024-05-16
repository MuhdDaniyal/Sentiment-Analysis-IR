import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Define constants
MAX_WORDS = 8000  # Vocabulary size
MAX_SEQUENCE_LENGTH = 500  # Max number of words per review

# Define function to load data from files
def load_data_from_files(directory):
    texts = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
    return texts

# Load training data
train_texts = load_data_from_files('train')  # Update the path as per your directory structure

# Load tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_texts)

# Load trained model
model = load_model('sentiment_analysis_model.h5')  # Provide the path to your trained model file

# Preprocess input review
def preprocess_review(review):
    sequence = tokenizer.texts_to_sequences([review])
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return sequence

# Make prediction
def predict_sentiment(review):
    sequence = preprocess_review(review)
    prediction = model.predict(sequence)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment

# Get user input
review = input("Enter your review: ")

# Predict sentiment
sentiment = predict_sentiment(review)
print("Predicted sentiment:", sentiment)
