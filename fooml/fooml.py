#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import collections as c


class Fooml(object):

    def __init__(self):
        self._out = sys.stdout
        self._err = sys.stderr
        self._ds = {}
        self._comp = c.OrderedDict()

    def add_dataset(self, ds, name='data'):
        if name in self._ds:
            print('Warning: Dataset with name "%s" already exists. Will be replaced' % name)
        self._ds[name] = ds

    def add_component(self, obj, name=None, func=None):
        self._comp[name] = obj

    def run(self):
        return


def main():
    return

if __name__ == '__main__':
    main()
