from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras import backend as K
import numpy as np

import os
import os.path


class SequenceImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png', normalize_seq=True,
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
        self.max_seq_length = -1
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
                if self.max_seq_length < curlen:
                    self.max_seq_length = curlen

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
        self.max_seq_length = val

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(self.max_seq_length * len(index_array) *
                           self.target_size[0] * self.target_size[1],
                           dtype=K.floatx())

        batch_x = batch_x.reshape((len(index_array), self.max_seq_length,
                                   self.target_size[0], self.target_size[1], 1))

        grayscale = self.color_mode == "grayscale"

        img = np.zeros(self.target_size[0] * self.target_size[1]) \
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
            curseqlen = self.max_seq_length if self.normalize_seq else len(seqimgs)
            for i2 in range(0, curseqlen):
                i2norm = int(i2 / (self.max_seq_length / len(seqimgs))) if self.normalize_seq else i2
                name = os.path.join(seqdir, seqimgs[i2norm])
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
