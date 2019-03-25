import os
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from itertools import product
import keras.backend as K
import tensorflow as tf
from functools import partial, update_wrapper

from process_data import load_dataset

K.set_session(
    K.tf.Session(
        config=K.tf.ConfigProto(
            device_count = {'GPU': 1 , 'CPU': 20}, 
            intra_op_parallelism_threads=32, 
            inter_op_parallelism_threads=32
        )
    )
)

MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100

def getLoss(weights, rnn=True):
    def w_categorical_crossentropy(y_true, y_pred):
        # print(y_true.shape, y_pred.shape)
        # y_true = K.print_tensor(y_true, message="y_true is: ")
        # y_pred = K.print_tensor(y_pred, message="y_pred is: ")

        nb_cl = len(weights)
        if(not rnn):
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, c_p] * K.cast(y_true, tf.float32)[:, c_t]  )
            return K.categorical_crossentropy(y_true, y_pred) * final_mask 
        else:
            final_mask = K.zeros_like(y_pred[:, :,0])
            y_pred_max = K.max(y_pred, axis=2)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], K.shape(y_pred)[1], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, :,c_p] * K.cast(y_true, tf.float32)[:, :,c_t]  )
            
            # final_mask = K.print_tensor(final_mask, message="final_mask is: ")
            # exit()
            # return K.categorical_crossentropy(y_pred, y_true)       
            return K.categorical_crossentropy(y_true, y_pred) * final_mask       
    return w_categorical_crossentropy

weights = np.ones((4,4))
for i in range(4):
    weights[1][i] = 5
    weights[2][i] = 3
    weights[3][i] = 3
weights[2][1] = 2

custom_loss = getLoss(weights)

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
        # model.compile(loss='categorical_crossentropy',
        model.compile(loss=custom_loss,
              optimizer='rmsprop', metrics=['acc'])
        self.model = model
        model.summary()


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


