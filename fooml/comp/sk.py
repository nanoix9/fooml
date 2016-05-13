#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
from fooml import dataset
from fooml.dt import slist
from fooml import util

class SkComp(comp.Comp):

    def __init__(self, obj):
        super(SkComp, self).__init__(obj)

    def trans(self, data):
        return self._obj.transform(data)

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

        #print X, y
        #self._obj.fit_transform(X, y)
        #print self._obj
        #return self._obj.fit_transform(X, y)

class Clf(SkComp):

    def __init__(self, obj):
        super(Clf, self).__init__(obj)

    def fit(self, data):
        X, y = data
        self._obj.fit(X, y)

    def trans(self, ds):
        X, y = ds
        if hasattr(self._obj, 'decision_function'):
            score = self._obj.decision_function(X)
        else:
            score = self._obj.predict_proba(X)
        #print '>>>>>', score
        #print '>>>>>', self._obj.classes_
        #sys.exit()
        # if it is a binary classification problem, return a 1-D array
        # of probablities of class 1
        #if score.shape[1] == 2:
        #    score = score[:,1]
        #    #print score
        #sys.exit()
        return score

    def fit_trans(self, data):
        X, y = data
        self.fit(data)
        score = self.trans(data)
        return dataset.dssy(score, y)

class Eva(SkComp):

    def __init__(self, obj):
        func, args, opt = obj
        super(Eva, self).__init__(func)
        self.args = args
        self.opt = opt

    def fit(self, data):
        pass

    def trans(self, ds):
        return None

    def fit_trans(self, data):
        eva_list = []
        for d in slist.iter_multi(data, strict=True):
            score, y = d
            eva = self._obj(y, score, *self.args, **self.opt)
            eva_list.append(eva)
        eva_str = str(eva)
        if '\n' not in eva_str:
            return dataset.desc('scores: ' + eva_str)
        else:
            return dataset.desc('scores:\n' + util.indent(eva_str))


def main():
    return

if __name__ == '__main__':
    main()
