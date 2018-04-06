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
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import TimeDistributed
from keras import metrics
from keras import backend as K

import socket
import struct
import numpy as np
import scipy.misc

import os
import os.path


class SequenceImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png', normalize_seq = True,
                 follow_links=False, interpolation='nearest'):

        self.normalize_seq = normalize_seq
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
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)

        self.samples = 0
        self.maxSeqLength = -1
        # salvar sequencias por classes
        self.sequences_per_class = []
        for class_name in self.class_indices:
            # salvando a quantidade de sequencias por classe = total samples

            cur_cls_dir = os.path.join(directory, class_name)

            seqdirname = sorted(os.listdir(cur_cls_dir))
            self.samples += len(seqdirname)

            self.sequences_per_class.append([os.path.join(cur_cls_dir, l)
                                             for l in os.listdir(cur_cls_dir)])
            for seqName in seqdirname:
                curlen = len(os.listdir(os.path.join(cur_cls_dir, seqName)))
                if self.maxSeqLength < curlen:
                    self.maxSeqLength = curlen

        print("found {} sequences belonging to {} classes".format(
              self.samples, len(self.class_indices)))

        self.classes = np.zeros((self.samples,), dtype='int32')
        sum_n = 0
        for k, v in self.class_indices.items():
            seqdirname = sorted(os.listdir(os.path.join(directory, k)))
            self.classes[sum_n:sum_n + len(seqdirname)] = np.full(len(seqdirname), v)
            sum_n += len(seqdirname)

        super(SequenceImageIterator, self).__init__(self.samples, batch_size,
                                                    shuffle, seed)

    def set_max_length(self, val):
        self.maxSeqLength = val

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(self.maxSeqLength * len(index_array) *
                           self.target_size[0] * self.target_size[1],
                           dtype=K.floatx())

        batch_x = batch_x.reshape((len(index_array), self.maxSeqLength,
                                   self.target_size[0], self.target_size[1], 1))

        grayscale = self.color_mode == "grayscale"

        img = np.zeros(self.target_size[0] * self.target_size[1])\
                .reshape((self.target_size[0], self.target_size[1]))

        for i1, j in enumerate(index_array):
            cls = 0
            cls_sum_aux = 0
            for it_cls_num in range(0, len(self.sequences_per_class)):
                if j > cls_sum_aux:
                    cls = it_cls_num
                cls_sum_aux += len(self.sequences_per_class[it_cls_num])

            sclsaux = sum([len(self.sequences_per_class[s1])
                           for s1 in range(0, cls)])
            curseq = j - sclsaux - 1

            seqdir = self.sequences_per_class[cls][curseq]

            last = -1
            seqimgs = sorted(os.listdir(seqdir), key=len)
            curseqlen = len(seqimgs) if self.normalize_seq else self.maxSeqLength
            for i2 in range(0, curseqlen):
                name = os.path.join(seqdir, seqimgs[i2])
                i2norm = int(i2 / (self.maxSeqLength / curseqlen )) if self.normalize_seq else i2
                if i2norm != last:
                    img = load_img(name, grayscale=grayscale,
                                   target_size=self.target_size,
                                   interpolation=self.interpolation)
                last = i2norm
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i1][i2] = x

        batch_y = np.zeros((len(index_array), len(self.class_indices)),
                            dtype=K.floatx())
        for it, label in enumerate(index_array):
            cls = 0
            cls_sum_aux = 0
            for it_cls_num in range(0, len(self.sequences_per_class)):
                if label > cls_sum_aux:
                    cls = it_cls_num
                cls_sum_aux += len(self.sequences_per_class[it_cls_num])

            batch_y[it, cls] = 1
        return batch_x, batch_y


# Fitting the CNN to the images


i_fold = 1
dirname     = '../Gestures/dynamic_poses/F1/train/'
testDirName = '../Gestures/dynamic_poses/F1/test'

seqIt = SequenceImageIterator(dirname, ImageDataGenerator(rescale=1./255),
                              target_size=(50, 50), color_mode='grayscale',
                              batch_size=64, class_mode='categorical')

testSeqIt = SequenceImageIterator(testDirName,
                                  ImageDataGenerator(rescale=1./255),
                                  target_size=(50, 50), color_mode='grayscale',
                                  batch_size=64, class_mode='categorical')

# Initialising the LSTM + CNN per Timestep


max_sequence_length = max(seqIt.maxSeqLength, testSeqIt.maxSeqLength)
testSeqIt.set_max_length(max_sequence_length)
seqIt.set_max_length(max_sequence_length)

classifier = Sequential()
classifier.add(TimeDistributed(Conv2D(32, (10, 10), input_shape=(50, 50, 1)
                               ,activation='relu'),
                               input_shape=(max_sequence_length, 50, 50, 1)))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Conv2D(32, (5, 5), input_shape=(25, 25, 1)
                                     ,activation='relu')))
classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
classifier.add(TimeDistributed(Flatten()))
classifier.add(LSTM(units=180, activation='tanh', return_sequences=True,
                    input_shape=(max_sequence_length, 25*25)))
classifier.add(LSTM(units=86, activation='tanh', return_sequences=True))
classifier.add(LSTM(units=34, activation='tanh'))
#classifier.add(LSTM(units=2, activation='softmax'))
classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy', metrics.categorical_accuracy])

classifier.summary()

# Fit the classifier
score = classifier.fit_generator(seqIt
                                 , steps_per_epoch=80
                                 , epochs=10
                                 , validation_data=testSeqIt
                                 , validation_steps=testSeqIt.samples/32)

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

