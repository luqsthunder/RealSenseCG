#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tvieira@ic.ufal.br
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed
from keras import metrics

import socket
import struct
import numpy as np
import scipy.misc

import os, os.path


class ImageSequenceDataGenerator:
    def __init__(self):
        self.imageGenerator = ImageDataGenerator(rescale=1./255)

    def flow_from_directory(self, directory):
        n = len([name for name in os.listdir(directory) if os.path.isdir(name)])
        output = []
        for it in range(0, n):
            images = self.imageGenerator.flow_from_directory(directory + str(it),
                                                             target_size=(50, 50),
                                                             color_mode='grayscale',
                                                             batch_size=32,
                                                             class_mode='categorical')
            output += images

        return output

# Initialising the CNN


classifier = Sequential()
classifier.add(TimeDistributed(Conv2D(8, (3, 3), input_shape=(50, 50, 1),
                               strides=(2, 2), activation='relu'),
                               input_shape=(30, 50, 50, 1)))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Conv2D(8, (3, 3), input_shape=(25, 25, 1),
                                      activation='relu')))
classifier.add(TimeDistributed(Flatten()))
classifier.add(LSTM(2, activation='relu', input_shape=(30, 25*25)))
#classifier.add(Dense(units=30, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy', metrics.categorical_accuracy])

# Part 2 - Fitting the CNN to the images

iFold = 1
train_datagen = ImageSequenceDataGenerator(rescale=1./255)

test_datagen = ImageSequenceDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('../cnn_hand_deep/F'
                                                 + str(iFold) + '/train')

test_set = test_datagen.flow_from_directory('../cnn_hand_deep/F'
                                            + str(iFold) + '/test')

# Fit the classifier


score = classifier.fit_generator(training_set,
                                 steps_per_epoch=40,
                                 epochs=25,
                                 validation_data=test_set,
                                 validation_steps=test_set.samples/32)

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

