#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import dataset
from dt import slist

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
        score = self._obj.predict_proba(X)
        return score

    def fit_trans(self, data):
        X, y = data
        self.fit(data)
        score = self.trans(X)
        return dataset.dssy(score, y)

class Eva(SkComp):

    def __init__(self, obj):
        func, args, opt = obj
        super(Eva, self).__init__(func)
        self.args = args
        self.opt = opt

    def fit(self, data):
        pass

    def trans(self, X):
        return None

    def fit_trans(self, data):
        eva_list = []
        for d in slist.iter_multi(data, strict=True):
            score, y = d
            eva = self._obj(y, score, *self.args, **self.opt)
            eva_list.append(eva)
        return dataset.desc(eva)


def main():
    return

if __name__ == '__main__':
    main()
