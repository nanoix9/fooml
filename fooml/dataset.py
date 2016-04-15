#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import sklearn.datasets as ds
import collections as c

ds_xy_t = c.namedtuple('ds_xy_t', 'X, y')
ds_train_test_xy_t = c.namedtuple('ds_train_test_xy_t', 'train, test')

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
        return ds_xy_t(ret['data'], ret['target'])

def main():
    return

if __name__ == '__main__':
    main()
