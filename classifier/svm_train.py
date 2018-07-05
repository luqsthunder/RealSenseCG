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
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# %% others
import socket
import struct

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import scipy.misc

from classifier_utilities.FrameSequenceIterator import SequenceImageIterator


i_fold = 1
dirname     = '../Gestures/dynamic_poses/Folders/F1/train/'
testDirName = '../Gestures/dynamic_poses/Folders/F1/test/'

seqIt = SequenceImageIterator(dirname, ImageDataGenerator(rescale=1./255),
                              target_size=(50, 50), color_mode='grayscale',
                              batch_size=32, class_mode='categorical',
                              normalize_seq=False, shuffle=False)

testSeqIt = SequenceImageIterator(testDirName,
                                  ImageDataGenerator(rescale=1./255),
                                  target_size=(50, 50), color_mode='grayscale',
                                  batch_size=32, class_mode='categorical',
                                  normalize_seq=False, shuffle = False)

# Initialising the LSTM + CNN per Timestep

max_sequence_length = max(seqIt.max_seq_length, testSeqIt.max_seq_length)
testSeqIt.set_max_length(max_sequence_length)
seqIt.set_max_length(max_sequence_length)

print(max_sequence_length)

# %%% SGD

train = np.arange(2100)
test = np.arange(900)

np.random.seed(5)
np.random.shuffle(train)

np.random.seed(5)
np.random.shuffle(test)

classifier = SGDClassifier(alpha=.0001, learning_rate='optimal', verbose=2)
# %%
last = 0
cls = np.arange(10)
for s in range(int(len(train) / 32)):
    curr = s * 32
    x, y = seqIt._get_batches_of_transformed_samples(train[last:curr])
    y = [np.argmax(l) for l in y]
    if(y == []):
        continue
    x = x.reshape((32, max_sequence_length*50*50))
    classifier.partial_fit(x, y, classes=cls)
    print('end a partial fit')
    last = curr

y_pred = []
y_true = []
last = 0
for s in range(int(len(test) / 32)):
    curr = s * 32
    x, y = testSeqIt._get_batches_of_transformed_samples(test[last:curr])
    if y.size == 0:
        continue

    x = x.reshape((32, max_sequence_length*50*50))
    y = [np.argmax(l) for l in y]
    y_true.extend(y)
    cur_pred = classifier.predict(x)
    y_pred.extend([p for p in cur_pred])
    last = curr

print(accuracy_score(y_true, y_pred))

# %% SVM classifier

y_all = []
x_all = []
last = 0
for s in range(1, int(len(train) / 32)):
    curr = s * 32
    x, y = seqIt._get_batches_of_transformed_samples(train[last:curr])
    if len(y) == 0:
        continue
    y = [np.argmax(k) for k in y]
    y_all.extend(y)
    x = x.reshape((32, max_sequence_length*50*50))
    x = [v for v in x]
    x_all.extend(x)
    print(curr)
    last = curr

y_all_test = []
x_all_test = []
last = 0
for s in range(1, int(len(test) / 32)):
    curr = s * 32
    x, y = testSeqIt._get_batches_of_transformed_samples(test[last:curr])
    if len(y) == 0:
        continue
    y = [np.argmax(k) for k in y]
    y_all_test.extend(y)
    x = x.reshape((32, max_sequence_length*50*50))
    x = [v for v in x]
    x_all_test.extend(x)
    print(curr)
    last = curr

classifier = SVC(C=1.0, kernel='rbf', gamma=.1)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format(.1, accuracy_score(y_all_test, y_pred)))

classifier = SVC(C=1.0, kernel='rbf', gamma=.5)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format('.5', accuracy_score(y_all_test, y_pred)))

classifier = SVC(C=1.0, kernel='rbf', gamma=1)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format('1', accuracy_score(y_all_test, y_pred)))

classifier = SVC(C=1.0, kernel='rbf', gamma=10)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format('10', accuracy_score(y_all_test, y_pred)))

classifier = SVC(C=1.0, kernel='rbf', gamma=100)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format('100', accuracy_score(y_all_test, y_pred)))

classifier = SVC(C=1.0, kernel='rbf', gamma=1000)
classifier.fit(x_all, y_all)
y_pred = classifier.predict(x_all_test)
print('gamma {} acc {}'.format('1000', accuracy_score(y_all_test, y_pred)))