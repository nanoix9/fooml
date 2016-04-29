#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp

class SkComp(comp.Comp):

    def __init__(self, obj):
        self._obj = obj

    def __repr__(self):
        return '%s(\n  obj=%s)' % (self.__class__.__name__, (str(self._obj)))

    def fit(self, data):
        X, y = data
        self._obj.fit(X, y)

    def trans(self, data):
        return self._obj.transform(data)

    def fit_trans(self, data):
        X, y = data
        #print X, y
        self._obj.fit_transform(X, y)
        print self._obj
        return self._obj.fit_transform(X, y)

class Sup(SkComp):
    pass

def main():
    return

if __name__ == '__main__':
    main()
