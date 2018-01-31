#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tvieira@ic.ufal.br
"""

# Importing the Keras libraries and packages


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

import socket
import struct
import numpy as np
import scipy.misc

# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
#classifier.add(Dense(units=100, activation='relu'))
classifier.add(Dense(units=14, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])

classifier.summary()

# Part 2 - Fitting the CNN to the images


iFold = 1
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
  'C:/Users/lucas/Documents/Projects/DL LSTM/cnn_bb/F1/train',
  target_size=(50, 50),
  color_mode='grayscale',
  batch_size=32,
  class_mode='categorical')

test_set = test_datagen.flow_from_directory(
  'C:/Users/lucas/Documents/Projects/DL LSTM/cnn_bb/F1/test',
  target_size=(50, 50),
  color_mode='grayscale',
  batch_size=32,
  class_mode='categorical',
  shuffle=False)

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

