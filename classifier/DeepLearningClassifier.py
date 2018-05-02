#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tvieira@ic.ufal.br
lucasthund3r@gmail.com
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import TimeDistributed
from keras import metrics

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
                                  normalize_seq=False)

# Initialising the LSTM + CNN per Timestep

max_sequence_length = max(seqIt.max_seq_length, testSeqIt.max_seq_length)
testSeqIt.set_max_length(max_sequence_length)
seqIt.set_max_length(max_sequence_length)

classifier = Sequential()
classifier.add(TimeDistributed(Conv2D(32, (10, 10), input_shape=(50, 50, 1),
                               activation='relu'),
                               input_shape=(max_sequence_length, 50, 50, 1)))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Conv2D(32, (5, 5), input_shape=(25, 25, 1),
                                      activation='relu')))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Flatten()))
classifier.add(LSTM(units=180, activation='tanh', return_sequences=True,
                    input_shape=(max_sequence_length, 25*25)))
classifier.add(LSTM(units=86, activation='tanh', return_sequences=True))
classifier.add(LSTM(units=34, activation='tanh'))
classifier.add(Dense(units=4, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy', metrics.categorical_accuracy])

classifier.summary()

# Fit the classifier
score = classifier.fit_generator(seqIt
                                 , steps_per_epoch=80
                                 , epochs=10
                                 , validation_data=testSeqIt
                                 , validation_steps=testSeqIt.samples/32)


# %% features for SVM



# %% SVM classifier



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

