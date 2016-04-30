#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp

class SkComp(comp.Comp):

    def __init__(self, obj):
        super(SkComp, self).__init__(obj)

    def __repr__(self):
        return '%s(\n  obj=%s)' % (self.__class__.__name__, (str(self._obj)))

    def trans(self, data):
        return self._obj.transform(data)

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

        #print X, y
        #self._obj.fit_transform(X, y)
        #print self._obj
        #return self._obj.fit_transform(X, y)

class Sup(SkComp):

    def __init__(self, obj):
        super(Sup, self).__init__(obj)

    def fit(self, data):
        X, y = data
        self._obj.fit(X, y)

    def trans(self, X):
        return self._obj.predict_proba(X)

    def fit_trans(self, data):
        X, y = data
        self.fit(data)
        return self.trans(X)

def main():
    return

if __name__ == '__main__':
    main()
