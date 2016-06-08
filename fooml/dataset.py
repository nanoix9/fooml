#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import sklearn.datasets as ds
import collections as c
import pandas as pd
import util

#dsxy = c.namedtuple('dsxy', 'X, y')
#dssy = c.namedtuple('dssy', 'score, y')

ds_train_test_xy_t = c.namedtuple('ds_train_test_xy_t', 'train, test')

class dataset(object):

    def __repr__(self):
        nvlist = [ nv for nv in util.getmembers(self)]
        return '<%s.%s>%s' % (self.__class__.__module__, \
                self.__class__.__name__, \
                nvlist)

class dsxy(dataset):

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def __iter__(self):
        yield self.X
        yield self.y

class dssy(dataset):

    def __init__(self, score=None, y=None):
        self.score = score
        self.y = y

    def __iter__(self):
        yield self.score
        yield self.y

class dscy(dataset):

    def __init__(self, cls=None, y=None):
        self.cls = cls
        self.y = y

    def __iter__(self):
        yield self.cls
        yield self.y

class dstv(dataset):

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def __iter__(self):
        yield self.train
        yield self.valid

class dslist(dataset):

    def __init__(self, *dss):
        self.dss = dss

    def __iter__(self):
        return iter(self.dss)

    def append(self, ds):
        self.dss.append(ds)

class desc(dataset):

    def __init__(self, desc=None):
        self.desc = desc

    def __repr__(self):
        return str(self.desc)

class Dataset(object):

    def __init__(self, name):
        self.name = name


class BasicDataset(Dataset):

    def __init__(self, name, data):
        super(BasicDataset, self).__init__(name)
        self.data = data

def load_data(name, **kwds):
    return load_toy(name, **kwds)

def load_csv(path, target, feature=None, dlm=','):
    ds = pd.read_csv(path)
    return dsxy

def load_image(image_path, target_path, sample_id, target=None):
    return

def load_toy(name, **kwds):
    name = name.lower()
    if name == 'iris':
        ret = ds.load_iris()
        return dsxy(ret['data'], ret['target'])
    elif name == 'digits':
        digits = ds.load_digits()
        n_samples = len(digits.images)
        if kwds.get('flatten', False):
            X = digits.images.reshape((n_samples, -1))
        else:
            X = digits.images
        y = digits.target
        return dsxy(X, y)
    elif name == 'mnist':
        from keras.datasets import mnist
        from keras.utils import np_utils
        nb_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        #print('X_train shape:', X_train.shape)
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        return dstv(dsxy(X_train, y_train), dsxy(X_test, y_test))

def _test_csv():
    path = 'test/min.csv'
    print load_csv(path, target='int')

def main():
    _test_csv()
    return

if __name__ == '__main__':
    main()
