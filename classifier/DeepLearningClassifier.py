#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tvieira@ic.ufal.br
lucasthund3r@gmail.com
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import TimeDistributed
from keras import metrics
from keras.callbacks import Callback
from keras import backend as K

import socket
import struct

import cv2 as cv

import numpy as np

import scipy.misc

from classifier_utilities.FrameSequenceIterator import SequenceImageIterator


# %% load data and run classifier

i_fold = 1
dirname     = '../Gestures/dynamic_poses/Folders/F1/train/'
testDirName = '../Gestures/dynamic_poses/Folders/F1/test/'

seqIt = SequenceImageIterator(dirname, ImageDataGenerator(rescale=1./255),
                              target_size=(50, 50), color_mode='grayscale',
                              batch_size=64, class_mode='categorical',
                              normalize_seq=False)

testSeqIt = SequenceImageIterator(testDirName,
                                  ImageDataGenerator(rescale=1./255),
                                  target_size=(50, 50), color_mode='grayscale',
                                  batch_size=64, class_mode='categorical',
                                  normalize_seq=False, shuffle=False)

# Initialising the LSTM + CNN per Timestep

max_seq_len = max(seqIt.max_seq_length, testSeqIt.max_seq_length)
testSeqIt.set_max_length(max_seq_len)
seqIt.set_max_length(max_seq_len)


class MaxWeights(Callback):
    def __init__(self):
        self.max_acc = -1
        self.acc_hist = []
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        curr_acc = logs['val_categorical_accuracy']
        if curr_acc > self.max_acc:
            self.max_acc = curr_acc
            self.weights = self.model.get_weights()
        self.acc_hist.append(curr_acc)


# %% classifier
def train_lstm(lstm_units, amount_filters, filters):
    classifier = Sequential()

    classifier.add(TimeDistributed(Conv2D(amount_filters, filters[0],
                                          input_shape=(50, 50, 1),
                                          activation='relu'),
                                   input_shape=(max_seq_len, 50, 50, 1)))
    classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    if len(filters) > 1:
        classifier.add(TimeDistributed(Conv2D(amount_filters, filters[1],
                                              activation='relu')))
        classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    classifier.add(TimeDistributed(Flatten()))

    classifier.add(LSTM(units=lstm_units[0], activation='tanh',
                        return_sequences=True if len(lstm_units) > 1 else False,
                        input_shape=(max_seq_len, 25 * 25)))
    if len(lstm_units) > 1:
        classifier.add(LSTM(units=lstm_units[1], activation='tanh',
                            return_sequences=True if len(lstm_units) > 2
                                             else False))
    if len(lstm_units) > 2:
        classifier.add(LSTM(units=lstm_units[2], activation='tanh'))

    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy', metrics.categorical_accuracy])

    classifier.summary()

    max_weights = MaxWeights()
    # Fit the classifier
    classifier.fit_generator(seqIt
                             , callbacks=[max_weights]
                             , steps_per_epoch=190
                             , epochs=15
                             , validation_data=testSeqIt
                             , validation_steps=testSeqIt.samples/64)

    classifier.set_weights(max_weights.weights)
    classifier.save('model_lstm{}_cnn{}.h5'.format(lstm_units, filters))
    classifier.save_weights('model_weights_lstm{}_cnn{}.h5'.format(lstm_units,
                                                                   filters))
    conf_mat = np.zeros((10, 10))
    for idx in range(900):
        spl = testSeqIt._get_batches_of_transformed_samples([idx])
        pred = classifier.predict(spl[0])
        cls = np.argmax(spl[1])
        classifier.reset_states()
        k = np.argmax(pred)
        conf_mat[cls, k] += 1

    out_str = 'lstm units {}\n' + \
              'best val categorical accuracy {}\n' + \
              'confusion mat for best epoch \n{}\n' + \
              'all accuracy per epoch \n{}\n\n'

    print(out_str.format(lstm_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist))
    print(out_str.format(lstm_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist), file=open('lstm64.txt', 'w'))


# %% cnn3D
def train_cnn3d(dense_units, cnn, filter_amount):
    classifier = Sequential()
    f0 = cnn[0]
    f1 = cnn[1]
    classifier.add(Conv3D(filter_amount, (f0, f0, f0),
                          input_shape=(max_seq_len, 50, 50, 1)))
    classifier.add(MaxPooling3D(pool_size=(2, 2, 2)))
    classifier.add(Conv3D(filter_amount, (f1, f1, f1),
                          input_shape=(max_seq_len, 50, 50, 1)))
    classifier.add(MaxPooling3D(pool_size=(2, 2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=dense_units[0], activation='relu'))
    if len(dense_units) > 1:
        classifier.add(Dense(units=dense_units[1], activation='relu'))
    if len(dense_units) > 2:
        classifier.add(Dense(units=dense_units[2], activation='relu'))
    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy', metrics.categorical_accuracy])

    classifier.summary()
    max_weights = MaxWeights()
#    Fit the classifier
    classifier.fit_generator(seqIt
                             , callbacks=[max_weights]
                             , steps_per_epoch=190
                             , epochs=5
                             , validation_data=testSeqIt
                             , validation_steps=testSeqIt.samples/64)

    classifier.set_weights(max_weights.weights)
    classifier.save('model_cnn3d{}_cnn{}_amount_' +
                    'filter{}.h5'.format(dense_units, cnn, filter_amount))
    classifier.save_weights('model_weights_cnn3d{}_cnn{}_' +
                            'amount_filter{}.h5'.format(dense_units, cnn))
    conf_mat = np.zeros((10, 10))
    for idx in range(900):
        spl = testSeqIt._get_batches_of_transformed_samples([idx])
        pred = classifier.predict(spl[0])
        cls = np.argmax(spl[1])
        classifier.reset_states()
        k = np.argmax(pred)
        conf_mat[cls, k] += 1

    out_str = 'lstm units {}\n' + \
              'best val categorical accuracy {}\n' + \
              'confusion mat for best epoch \n{}\n' + \
              'all accuracy per epoch \n{}\n\n'

    print(out_str.format(dense_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist))
    print(out_str.format(dense_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist), file=open('cnn3D64.txt', 'w'))

train_cnn3d([300, 200, 100], [3, 3], 64)
K.clear_session()
train_lstm([100, 100], 64, [[3, 3], [3, 3]])
K.clear_session()


# %%%

# %%%%%%%  SERVIDOR  %%%%%%%%%%%%%%

TCP_IP = '127.0.0.1'
TCP_PORT = 31000
BUFFER_SIZE = 50*3
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print('Waiting connection…')
conn, addr = s.accept() #CONECTOU
print('Connection address:', addr)

while 1: # WHILE INFINITO PARA SEMPRE ESTAR RECEBENDO IMAGENS
    current_size = 0
    size = 50*50*4 # width * heigth * sizeof(unsigned int) #1228800
    buffer = b""

    contLines = 0

    while current_size < size: #WHILE PARA RECEBER UMA IMAGEM, UMA LINHA DE CADA VEZ
        data = conn.recv(BUFFER_SIZE)
        #print(len(data))
        if not data:
            break
        if len(data) + current_size > size:
            data = data[:size-current_size]
        #conn.send('ok'.encode())
        buffer += data
        current_size += len(data)

    # CONVERTE A IMAGEM PARA O FORMATO DO KERAS
    imgint=struct.unpack("2500I", buffer)   #"307200I",buffer)

    npimg=np.array(imgint).reshape(50,50)
    sciimg = scipy.misc.toimage(npimg)
    imglow=sciimg.resize((50,50))
    npimglow = np.array(imglow).reshape(1,50,50,1)
    # PREDICT DA IMAGEM
    predvec = classifier.predict(npimglow)

    predvecstr = "%f" % predvec[0][0]
    for i in range(1, len(predvec[0])):
        predvecstr = "%s %f" % (predvecstr,predvec[0][i])
    conn.send(predvecstr.encode())
