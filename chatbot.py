""" Chatbot based on Recurrent Neural Nets and trained on OpenSubtitles. 
Original inspiration from: http://arxiv.org/pdf/1506.05869v1.pdf

Usage:
    chatbot.py <FILE>

Options:
    <FILE>      File containing parsed subtitles (e.g., data/lacollectioneuse.txt)
"""
import json
from docopt import docopt
from numpy import vstack, where

from keras.models import Sequential, model_from_json
from keras.optimizers import Adagrad
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import text

args = docopt(__doc__)

max_features = 512

tokenizer = text.Tokenizer(max_features) # keep top-1000 words

fh = open(args["<FILE>"], 'ro')
tokenizer.fit_on_texts(fh)

fh.seek(0) # reset the file pointer
X = tokenizer.texts_to_matrix(fh)
fh.close()

X_train = vstack((X[1:], X[:-1]))
Y_train = vstack((X[:-1], X[1:]))

print len(where(X_train[0] == 1)[0])
sys.exit
try:
    fh = open('keras.model.json', 'rb')
    model = model_from_json(fh.read())
    fh.close()
except:
    model = Sequential()
    # Add a mask_zero=True to the Embedding connstructor if 0 is a left-padding value in your data
    model.add(Embedding(max_features, 64))
    model.add(LSTM(64, 64, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(TimeDistributedDense(64, 64, activation='sigmoid',))
    model.add(Dropout(0.5))
    model.add(LSTM(64, 64, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(64, max_features))
    model.add(Activation('sigmoid'))
    

    adagrad = Adagrad(lr=0.01, epsilon=1e-6, clipnorm=1.)
    model.compile(loss='binary_crossentropy', optimizer=adagrad)

    fh = open('keras.model.json', 'wb')
    fh.write(model.to_json())
    fh.close()

model.fit(X_train, Y_train, batch_size=16, nb_epoch=3, verbose=1.)

print model.predict(X[:2], verbose=1.)

# score = model.evaluate(X_test, Y_test, batch_size=16)

