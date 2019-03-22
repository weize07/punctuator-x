#!encoding=utf-8
import argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed

import numpy as np

data_path = '../data/cornell-movie/movie_lines.txt'
filtered_path = '../data/cornell-movie/movie_lines_filtered.txt'
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 30
categories = {
    ' ': 0,
    '.': 1,
    '?': 2,
    ',': 3,
    '!': 1,
}

def filter_sentences():
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as in_file, open(filtered_path, 'w') as out_file:
        for line in in_file:
            pos = line.rfind('+++$+++ ')
            if pos == -1:
                continue
            out_file.write(line[pos + 8:])
            # print(line)
            # print(line[pos + 8:])
            # break

def load_dataset():
    X = []
    Y = []
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    with open(filtered_path, 'r') as f:
        texts = []
        count = 0
        for line in f:
            count += 1
            if count > 10000:
                break
            if not line.strip():
                continue
            texts.append(line)
            current = 'a'
            is_alnum = True
            y = []
            for i in range(len(line)):
                if line[i].isalnum() or line[i] in ['\"', '\'']:
                    if is_alnum:
                        current = line[i]
                    else:
                        pun = current.strip()
                        if pun not in categories:
                            y.append(0)
                        else:
                            y.append(categories[pun])
                        current = line[i]
                        is_alnum = True
                else:
                    if not is_alnum:
                        current += line[i]
                    else:
                        current = line[i]
                        is_alnum = False
            if not is_alnum:
                pun = current.strip()
                if pun not in categories:
                    y.append(0)
                else:
                    y.append(categories[pun])
                current = line[i]
            else:
                y.append(0)

            Y.append(to_categorical(np.array(y), num_classes=4))
            # Y.append(y)

        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        # padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        ne_count = 0
        # reverse_index = { tokenizer.word_index[key]: key for key in tokenizer.word_index }
        filterd_X = []
        filterd_Y = []
        for i in range(len(sequences)):
            if len(sequences[i]) == len(Y[i]):
                ne_count += 1
            else:
                filterd_X.append(sequences[i])
                filterd_Y.append(Y[i])
        print('ne percentage: %d/%d' % (ne_count, len(texts)))

        X = pad_sequences(np.array(filterd_X), maxlen=MAX_SEQUENCE_LENGTH)
        Y = pad_sequences(np.array(filterd_Y), maxlen=MAX_SEQUENCE_LENGTH)
        # Y = np.reshape(Y, (len(Y), len(Y[0]), 1))
        return (X, Y, tokenizer)
    

if __name__ == '__main__':
    # filter_sentences()
    X, Y, tokenizer = load_dataset()
    print(X.shape)
    print(Y.shape)
    model = Sequential()
    embedding = Embedding(len(tokenizer.word_index) + 1,
                                    100,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedded_sequences = embedding(sequence_input)

    model.add(sequence_input)
    model.add(embedding)

    # input_array = np.random.randint(1000, size=(32, 10))

    model.compile('rmsprop', 'mse')
    output_array = model.predict(X)
    print(output_array.shape)

