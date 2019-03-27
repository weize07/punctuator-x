import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model


from process_data import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUMS_OF_GPU = 0
NUMS_OF_CPU = 4

MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 100

class Puntuator:
    def __init__(self, glove_dir, tokenizer):
        self.glove_path = os.path.join(glove_dir, 'glove.6B.100d.txt')
        self.tokenizer = tokenizer
        self.load_glove_layer()

    def load_glove_layer(self):
        embeddings_index = {}
        word_index = self.tokenizer.word_index
        f = open(self.glove_path, encoding='utf-8', errors='ignore')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)


    def save(self):
        return

    def train(self, X, Y, epochs=10, batch_size=32):
        # padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        # Y = pad_sequences(Y, maxlen=MAX_SEQUENCE_LENGTH)
        # print(padded.shape)
        # print(Y.shape)
        # print(padded[0])
        # print(Y[0])
        print(X.shape)
        print(Y.shape)

        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, X, batch_size):
        # padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        return self.model.predict(X, batch_size)

    def load(self, model_path):
        return

    def build_model(self):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(LSTM(EMBEDDING_DIM, return_sequences=True))
        # model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))

        model.add(TimeDistributed(Dense(4, activation='softmax')))
        if NUMS_OF_GPU > 1:
            pal_model = multi_gpu_model(model, NUMS_OF_GPU)
        else:
            pal_model = model
        pal_model.compile(loss='categorical_crossentropy',
        # pal_model.compile(loss=custom_loss,
              optimizer='rmsprop', metrics=['acc'])
        self.model = pal_model
        pal_model.summary()


if __name__ == '__main__':
    X, Y, tokenizer = load_dataset()
    puntuator = Puntuator('../data', tokenizer)
    puntuator.build_model()
    # puntuator.train(X, Y, 1, 32)
    puntuator.train(X, Y)
    tests = [
        "Hi nice to meet you",
        "Oh that's impossible"
    ]
    testX = tokenizer.texts_to_sequences(tests)
    testX = pad_sequences(testX, maxlen=MAX_SEQUENCE_LENGTH)
    print(testX)
    # print(testX.shape)
    print(puntuator.predict(testX, len(tests)))
