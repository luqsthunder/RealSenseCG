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
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed
from keras import metrics
from keras import backend as K

import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

import socket
import struct
import numpy as np
import scipy.misc

import os, os.path

class SequenceImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest'):

        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.classes = classes
        self.class_mode = class_mode
        self.interpolation = interpolation

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)

        # contar os diretorios de classes e armazenar nome dos diretorios
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)

        self.samples = 0

        # salvar sequencias por classes
        self.sequencesPerClass = []
        self.sequencesImgsPerClass = []
        for className in self.class_indices:
            # salvando a quantidade de sequencias por classe = total samples
            seqDirNames = sorted(os.listdir(directory + '/' + className))
            self.samples += len(seqDirNames)
            self.sequencesPerClass.append(seqDirNames)
            seqClassImgs = []
            # lendo images para cada sequencia
            for seqName in seqDirNames:
                self.classes
                seqClassImgs.append(sorted(os.listdir(directory + '/' +
                                                      className + '/' +
                                                      seqName), key=len))
            self.sequencesImgsPerClass.append(seqClassImgs)

        print("found %d sequences belonging to %d classes",
              self.samples, len(self.class_indices))

        self.classes = np.zeros((self.samples,), dtype='int32')
        sumN = 0
        for k, v in self.class_indices.items():
            seqDirNames = sorted(os.listdir(directory + '/' + k))
            self.classes[sumN:sumN+len(seqDirNames)] = np.full(len(seqDirNames),v)
            sumN += len(seqDirNames)


        super(SequenceImageIterator, self).__init__(self.samples, batch_size,
                                                    shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(30 * self.batch_size * 50 * 50, dtype=K.floatx())
        batch_x = batch_x.reshape((self.batch_size, 30, 50, 50, 1))

        print(len(self.class_indices))

        grayscale = self.color_mode == "grayscale"

        for i1, j in enumerate(index_array):
            cls = int(j / self.batch_size)
            curSeq = j % self.batch_size
            for i2, it in enumerate(self.sequencesImgsPerClass[cls][curSeq]):
                name = self.directory + "/P" + str(cls + 1) + "/" + self.sequencesPerClass[cls][curSeq] + "/" + it
                img = load_img(name, grayscale=grayscale,
                               target_size=self.target_size,
                               interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i1][i2] = x

        batch_y = np.zeros((self.batch_size, len(self.class_indices)),
                            dtype=K.floatx())
        for it, label in enumerate(self.classes[index_array]):
            batch_y[it, label] = 1
        return batch_x, batch_y



# Initialising the LSTM + CNN per Timestep


classifier = Sequential()
classifier.add(TimeDistributed(Conv2D(8, (3, 3), input_shape=(50, 50, 1),
                               strides=(2, 2), activation='relu'),
                               input_shape=(30, 50, 50, 1)))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Conv2D(8, (3, 3), input_shape=(25, 25, 1),
                                      activation='relu')))
classifier.add(TimeDistributed(Flatten()))
classifier.add(LSTM(2, activation='softmax', input_shape=(30, 25*25)))
#classifier.add(Dense(units=30, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy', metrics.categorical_accuracy])

classifier.summary()

# Part 2 - Fitting the CNN to the images

iFold = 1
dirname = '../Gestures/dynamic_poses/F1/train'

seqIt = SequenceImageIterator(dirname, ImageDataGenerator(rescale=1./255),
                              target_size=(50, 50), color_mode='grayscale',
                              batch_size=32, class_mode='categorical')

# Fit the classifier


score = classifier.fit_generator(seqIt,
                                 steps_per_epoch=40
                                 ,epochs=25)
                                 #,validation_data=test_set
                                 #,validation_steps=test_set.samples/32)

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

