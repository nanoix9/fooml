#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
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

    def __init__(self, X, y=None, index=None):
        self.X = X
        self.y = y
        self.index = index

    def __iter__(self):
        yield self.X
        yield self.y

    def nsamples(self):
        return len(self.X)

    def get_index(self):
        raise NotImplementedError()
        if isinstance(self.X, (pd.DataFrame, pd.Series)):
            return self.X.index
        else:
            return self.index

class dssy(dataset):

    def __init__(self, score, y=None, index=None):
        self.score = score
        self.y = y
        self.index = index

    def __iter__(self):
        yield self.score
        yield self.y

    def nsamples(self):
        return len(self.score)

class dscy(dataset):

    def __init__(self, cls, y=None, index=None):
        self.cls = cls
        self.y = y
        self.index = index

    def __iter__(self):
        yield self.cls
        yield self.y

    def nsamples(self):
        return len(self.cls)

class dstv(dataset):

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def __iter__(self):
        yield self.train
        yield self.valid

    def nsamples(self):
        return self.train.nsamples(), self.valid.nsamples()

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
        return util.joins(self.desc, sep='\n', ind=2)

class Dataset(object):

    def __init__(self, name):
        self.name = name


class BasicDataset(Dataset):

    def __init__(self, name, data):
        super(BasicDataset, self).__init__(name)
        self.data = data

def load_data(name, **kwds):
    return load_toy(name, **kwds)

def load_csv(path, index_col=None, target=None, feature=None, **kwds):
    df = pd.read_csv(path, index_col=index_col, encoding='utf8', **kwds)
    #if index_col:
    #    index = np.array(df.index)
    #else:
    #    index = None
    index = None
    return dsxy(df, None, index=index)

def load_image(image_path, target_path, sample_id, target=None):
    return

def _subdirs(path):
    _, dirnames, _ = next(os.walk(path), (None, [], None))
    #print path
    #print [x for x in os.walk(path)]
    return dirnames

def get_im_cv2(path, resize=None, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    if resize:
        img = cv2.resize(img, resize)
    return img

def load_image_grouped(image_path, resize=None, color_type=1, file_ext=None, **_):
    ''' Images are stored by class, i.e. images for the same class are put in the same directory.
    load all images in all subdirectory and treat the name of subdirectory as class name. '''

    X_train = []
    y_train = []
    file_names = []

    logger.info('read images from path "%s"' % image_path)
    for j in _subdirs(image_path):
        if file_ext:
            pattern = '*.%s' % file_ext
        else:
            pattern = '*'
        path = os.path.join(image_path, str(j), pattern)
        files = glob.glob(path)
        logger.info('load folder {}: {} files'.format(j, len(files)))
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, resize, color_type=color_type)
            X_train.append(img)
            y_train.append(j)
            #file_names.append(fl)
            file_names.append(flbase)
            #driver_id.append(driver_data[flbase])

    #unique_drivers = sorted(list(set(driver_id)))
    #print('Unique drivers: {}'.format(len(unique_drivers)))
    #print(unique_drivers)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return dsxy(X_train, y_train, index=np.array(file_names))

def load_image_flat(image_path, target=None, resize=None, color_type=1, file_ext=None, **_):
    ''' load all images in a directory'''

    X_train = []
    #y_train = []
    y_train = None
    file_names = []

    if file_ext:
        pattern = '*.%s' % file_ext
    else:
        pattern = '*'
    path = os.path.join(image_path, pattern)
    files = glob.glob(path)
    logger.info('read images from path {}: {} files'.format(path, len(files)))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, resize, color_type=color_type)
        X_train.append(img)
        #y_train.append(j)
        file_names.append(flbase)

    #X_train = np.array(X_train)
    X_train = np.array(X_train)
    #y_train = np.array(y_train)

    return dsxy(X_train, y_train, index=np.array(file_names))

def load_image_patt(image_path, feature_pattern, get_target=None, resize=None,
            color_type=1, target_color_type=None, **_):
    ''' Both feature and target are images.
    Load images matching patterns in a directory.
    That matches `feature_pattern` are treat as features,
    and the file name of corresponding target are computed from `get_target`.
    If `get_target` is not given, then only load features;
    If `get_target` is given but cannot found the target file for one feature,
    then such sample will be discarded.'''

    feature_regex = re.compile(feature_pattern)
    feature_color_type = color_type
    if target_color_type is None:
        target_color_type = color_type
    X_train = []
    y_train = []
    file_names = []

    path = os.path.join(image_path, '*')
    files = glob.glob(path)
    #print files
    logger.info('read images from path {}: {} files'.format(path, len(files)))
    for fl in files:
        flbase = os.path.basename(fl)
        if feature_regex.match(flbase):
            target_file = os.path.join(path, get_target(flbase))
            if not os.path.isfile(target_file):
                continue
        img_feat = get_im_cv2(fl, resize, color_type=feature_color_type)
        img_targ = get_im_cv2(fl, resize, color_type=target_color_type)
        X_train.append(img_feat)
        y_train.append(img_targ)
        file_names.append(flbase)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return dsxy(X_train, y_train, index=np.array(file_names))


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

def save_csv(ds, path, opt):
    #print ds
    if isinstance(ds, dssy):
        #print 'score:', ds.score
        columns = _get_opt_lazy(opt, 'columns', False)
        label = _get_opt_lazy(opt, 'label', False)
        pd.DataFrame(ds.score, columns=columns, index=ds.index).to_csv(path, index_label=label)
    else:
        raise TypeError('Type not supported: {}'.format(util.get_type_fullname(ds)))

def _get_opt_lazy(opt, key, default):
    if key in opt:
        opt_val = opt[key]
        if hasattr(opt_val, '__call__'):
            opt_val = opt_val()
        return opt_val
    else:
        return default

def map(func, data):
    if not isinstance(data, dataset):
        raise TypeError('data set is not an instance of dataset')

    dtran = clazz = data.__class__()
    for name, value in util.getmembers(data):
        #print '>>>', name, value
        setattr(dtran, name, func(name, value))
    return dtran

def mapx(func, data):
    if isinstance(data, list):
        return [mapx(func, d) for d in data]
    elif isinstance(data, dsxy):
        return dsxy(func(data.X), data.y, data.index)
    elif isinstance(data, dstv):
        return dstv(mapx(func, data.train), mapx(func, data.valid))
    else:
        raise TypeError('not supported data type: %s' % data.__class__)

def mapy(func, data):
    if isinstance(data, list):
        return [mapy(func, d) for d in data]
    elif isinstance(data, dsxy):
        if data.y is None:
            return data
        else:
            return dsxy(data.X, func(data.y), data.index)
    elif isinstance(data, dscy):
        return dscy(func(data.c), __apply_maybe(func, data.y))
    elif isinstance(data, dstv):
        return dstv(mapy(func, data.train), mapy(func, data.valid))
    else:
        raise TypeError('not supported data type: %s' % data.__class__)

def mapxy(func, data):
    pass

def mergex(func, data):
    if not isinstance(data, list):
        return mapx(func, data)
    ds_main = data[0]
    Xs = [d.X for d in data]
    X_new = func(*Xs)
    return dsxy(X_new, ds_main.y, ds_main.index)

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

def split(data, it, iv):
    if not isinstance(data, dsxy):
        raise TypeError()
    X, y = data
    index = data.index
    if index is None:
        index = np.arange(X.shape[0])
    Xt = X[it]; yt = y[it]; tidx = index[it]
    Xv = X[iv]; yv = y[iv]; vidx = index[iv]
    return dstv(dsxy(Xt, yt, tidx), dsxy(Xv, yv, vidx))


######### Tests ########

def _test_csv():
    path = 'test/min.csv'
    print load_csv(path, target='int')

def main():
    _test_csv()
    return

if __name__ == '__main__':
    main()
