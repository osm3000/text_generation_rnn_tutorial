"""
This is a direct transformation from the model implemented in keras
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""
import utils
import numpy as np
import models
import torch
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.utils import shuffle
import sys
import random

# Load the data - cropus
with open('nietzsche.txt', "r") as fileHandle:
    text = fileHandle.read().lower()
print('\ncorpus length:', len(text))

# Get the unique characters in the corpus
chars = sorted(list(set(text)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in sequences (windows) of 'maxlen' characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# For each sequence/window --> x, chose the next character to predict --> y
# Also, but the data into one hot encoding (onehot encoding can be see as probability distribution. The sum of all the elements equal one)
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)))
y = np.zeros((len(sentences)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1.0
    y[i] = char_indices[next_chars[i]]
print ("x.shape: {}, y.shape: {}".format(x.shape, y.shape))
# corpus length: 600901
# total chars: 59
# nb sequences: 200287
# Vectorization...
# x.shape: (200287, 40, 59), y.shape: (200287,)

# Declare the model --> refer to the file models.py for details about this model.
rnn_model = models.SimpleGRU(in_size=len(chars), out_size=len(chars), hidden_size=200)
if torch.cuda.is_available():
    rnn_model.cuda()
optimizer = Adam(rnn_model.parameters(), lr=0.005)
loss_funtion = torch.nn.NLLLoss() # Since I am using LogSoftmax, I use NLLLoss function

batch_size = 2048 # How much data to process in parallel
train_mode = False # If you want to train and store models --> Not using them for generation, yet
nb_epochs = 150
if train_mode:
    for epoch in range(nb_epochs):
        loss_values = []
        rnn_model.train()
        for batch, i in enumerate(range(0, x.shape[0] - 1, batch_size)):
            # X_batch, y_batch = utils.getbatch(x, y, i=i, batch_size=batch_size)
            X_batch, y_batch = shuffle(x, y, n_samples=batch_size) # Dawood's idea for a randomly-sampled batch

            X_batch = Variable(torch.from_numpy(X_batch).float(), requires_grad=False)
            y_batch = Variable(torch.from_numpy(y_batch).long(), requires_grad=False)
            if torch.cuda.is_available():
                X_batch.cuda()
                y_batch.cuda()

            model_output = rnn_model.forward(X_batch)

            optimizer.zero_grad()
            loss = loss_funtion(model_output, y_batch)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.data[0])

            del X_batch
            del y_batch

        if (epoch+1) % 5 == 0: # I will save the model in this case
            torch.save(rnn_model.state_dict(), "rnn_model_epoch_" + str(epoch))

        print ("Epoch: {}, Loss: {}".format(epoch, np.mean(loss_values)))
else:
    # start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 47
    for i in [139]: # Just use the model trained after 139 epochs
        rnn_model.load_state_dict(torch.load("rnn_model_epoch_" + str(i)))
        rnn_model.eval()

        print ("After {} of epochs".format(i))
        """
        If the temperature is too low (near zero), this amounts to argmax selection
        If the temperature is too high, this amounts to uniform random selection
        We usually want something in the middle (usually in the range of 0.5 -> 1)
        """
        for temperature in [0.01, 0.5, 3.0]:
            print('----- temperature:', temperature)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.0
                x_pred = Variable(torch.from_numpy(x_pred).float())
                if torch.cuda.is_available():
                    x_pred.cuda()

                preds = rnn_model.forward(x_pred)
                preds = preds.data.cpu().numpy()[0]
                next_index = utils.sample(preds, temperature)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            print ("Generated sentence: ", generated)
            print ("/*"*100)
            print ("\n"*3)
