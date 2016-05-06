#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import sklearn.datasets as ds
import collections as c

#dsxy = c.namedtuple('dsxy', 'X, y')
#dssy = c.namedtuple('dssy', 'score, y')

ds_train_test_xy_t = c.namedtuple('ds_train_test_xy_t', 'train, test')

class dsxy(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __iter__(self):
        yield self.X
        yield self.y

class dssy(object):

    def __init__(self, score, y):
        self.score = score
        self.y = y

    def __iter__(self):
        yield self.score
        yield self.y

class desc(object):

    def __init__(self, desc):
        self.cnt = desc

    def __repr__(self):
        return str(self.cnt)

class Dataset(object):

    def __init__(self, name):
        self.name = name


class BasicDataset(Dataset):

    def __init__(self, name, data):
        super(BasicDataset, self).__init__(name)
        self.data = data

def load_data(name):
    return load_toy(name)

def load_toy(name):
    name = name.lower()
    if name == 'iris':
        ret = ds.load_iris()
        return dsxy(ret['data'], ret['target'])

def main():
    return

if __name__ == '__main__':
    main()
