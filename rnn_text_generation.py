"""
This is a direct transformation from the model implemented in keras
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""
import utils
from keras.utils.data_utils import get_file
import numpy as np
import models
import torch
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from sklearn.utils import shuffle

# Extras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random
import sys

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('\ncorpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)))
# y = np.zeros((len(sentences), len(chars)))
y = np.zeros((len(sentences)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1.0
    # y[i, char_indices[next_chars[i]]] = 1.0
    y[i] = char_indices[next_chars[i]]
print ("x.shape: {}, y.shape: {}".format(x.shape, y.shape))

# PyTorch part
rnn_model = models.SimpleGRU(in_size=len(chars), out_size=len(chars), hidden_size=128).cuda()
optimizer = Adam(rnn_model.parameters(), lr=0.01)
loss_funtion = torch.nn.CrossEntropyLoss()

batch_size = 1024
for epoch in range(40):
    loss_values = []
    for batch, i in enumerate(range(0, x.shape[0] - 1, batch_size)):
        # X_batch, y_batch = utils.getbatch(x, y, i=i, batch_size=batch_size)
        X_batch, y_batch = shuffle(x, y, n_samples=batch_size)

        X_batch = Variable(torch.from_numpy(X_batch).float(), requires_grad=False).cuda()
        y_batch = Variable(torch.from_numpy(y_batch).long(), requires_grad=False).cuda()

        model_output = rnn_model.forward(X_batch)

        optimizer.zero_grad()
        loss = loss_funtion(model_output, y_batch)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.data[0])

        del X_batch
        del y_batch
        # print ("i: ", i)
    print ("Epoch: {}, Loss: {}".format(epoch, np.mean(loss_values)))
    # exit()
# Keras part
# build the model: a single LSTM
# print('Build model...')
# model = Sequential()
# model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))
#
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
