import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
from data_utils import train_texts

# Define constants
MAX_WORDS = 8000  # Vocabulary size
MAX_SEQUENCE_LENGTH = 500  # Max number of words per review
EMBEDDING_DIM = 100  # Dimension of word embeddings
BATCH_SIZE = 128
NUM_EPOCHS = 15

# Load data
# Get the absolute path of the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Construct paths to train and test folders
train_dir = os.path.join(current_dir, 'train')
test_dir = os.path.join(current_dir, 'test')


train_texts = []
train_labels = []

for category in ['neg', 'pos']:
    train_path = os.path.join(train_dir, category)
    for fname in os.listdir(train_path):
        if fname.endswith('.txt'):
            with open(os.path.join(train_path, fname), 'r', encoding='utf-8') as f:
                train_texts.append(f.read())
            train_labels.append(0 if category == 'neg' else 1)

test_texts = []
test_labels = []

for category in ['neg', 'pos']:
    test_path = os.path.join(test_dir, category)
    for fname in os.listdir(test_path):
        if fname.endswith('.txt'):
            with open(os.path.join(test_path, fname), 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
            test_labels.append(0 if category == 'neg' else 1)

# Tokenize texts
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Define CNN model
model = Sequential()
model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Train model with early stopping
model.fit(train_data, np.array(train_labels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_split=0.2)

loss, accuracy = model.evaluate(test_data, np.array(test_labels))
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
# Save the trained model to a file
model.save('sentiment_analysis_model.h5')
