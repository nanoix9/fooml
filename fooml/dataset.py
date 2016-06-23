#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
import glob
import sklearn.datasets as ds
import collections as c
import numpy as np
import pandas as pd
import util
from log import logger

try:
    import cv2
except ImportError:
    logger.warning('load cv2 failed')

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

def _subdirs(path):
    _, dirnames, _ = next(os.walk(path), (None, [], None))
    #print path
    #print [x for x in os.walk(path)]
    return dirnames

def _get_im_cv2(path, resize=None, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    if resize:
        img = cv2.resize(img, resize)
    return img

def load_image_grouped(image_path, resize=None, file_ext='jpg'):
    X_train = []
    y_train = []

    logger.info('read images from path "%s"' % image_path)
    for j in _subdirs(image_path):
        path = os.path.join(image_path, str(j), '*.%s' % file_ext)
        files = glob.glob(path)
        logger.info('load folder {}: {} files'.format(j, len(files)))
        for fl in files:
            flbase = os.path.basename(fl)
            img = _get_im_cv2(fl, resize, color_type=1)
            X_train.append(img)
            y_train.append(j)
            #driver_id.append(driver_data[flbase])

    #unique_drivers = sorted(list(set(driver_id)))
    #print('Unique drivers: {}'.format(len(unique_drivers)))
    #print(unique_drivers)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return dsxy(X_train, y_train)

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

        #X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        #X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        #X_train = X_train.astype('float32')
        #X_test = X_test.astype('float32')
        #X_train /= 255
        #X_test /= 255
        #print('X_train shape:', X_train.shape)
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        #y_train = np_utils.to_categorical(y_train, nb_classes)
        #y_test = np_utils.to_categorical(y_test, nb_classes)
        return dstv(dsxy(X_train, y_train), dsxy(X_test, y_test)), dsxy(X_test, y_test)

def map(func, data):
    if not isinstance(data, dataset):
        raise TypeError('data set is not an instance of dataset')

    dtran = clazz = data.__class__()
    for name, value in util.getmembers(data):
        #print '>>>', name, value
        setattr(dtran, name, func(name, value))
    return dtran

def mapx(func, data):
    if isinstance(data, dsxy):
        return dsxy(func(data.X), data.y)
    elif isinstance(data, dstv):
        return dstv(mapx(func, data.train), mapx(func, data.valid))
    else:
        raise TypeError('not supported data type: %s' % data.__class__)
def mapy(func, data):
    if isinstance(data, dsxy):
        return dsxy(data.X, __apply_maybe(func, data.y))
    elif isinstance(data, dscy):
        return dscy(func(data.c), __apply_maybe(func, data.y))
    elif isinstance(data, dstv):
        return dstv(mapy(func, data.train), mapy(func, data.valid))
    else:
        raise TypeError('not supported data type: %s' % data.__class__)

def mapxy(func, data):
    pass

def __apply_maybe(func, data):
    if data is None:
        return None
    else:
        return func(data)

def get_train_valid(data):
    if isinstance(data, dsxy):
        X_train, y_train = data
        X_valid, y_valid = None, None
    elif isinstance(data, dstv):
        (X_train, y_train), ds_valid = data
        if ds_valid is None:
            X_valid, y_valid = None, None
        else:
            X_valid, y_valid = ds_valid
    else:
        raise TypeError('Unknown dataset type: %s' % data.__class__)
    return X_train, y_train, X_valid, y_valid

def get_train(data):
    if isinstance(data, dsxy):
        X_train, y_train = data
    elif isinstance(data, dstv):
        (X_train, y_train), ds_valid = data
    else:
        raise TypeError('Unknown dataset type: %s' % data.__class__)
    return X_train, y_train



######### Tests ########

def _test_csv():
    path = 'test/min.csv'
    print load_csv(path, target='int')

def main():
    _test_csv()
    return

if __name__ == '__main__':
    main()
