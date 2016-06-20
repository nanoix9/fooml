#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import comp
import mixin
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger

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

class TargTrans(mixin.TargTransMixin, SkComp):

    def __init__(self, obj):
        super(TargTrans, self).__init__(obj)
        self._fit_func = self._obj.fit
        self._fit_trans_func = self._obj.fit_transform
        self._trans_func = self._obj.transform

class TargInvTrans(mixin.TargTransMixin, SkComp):

    def __init__(self, another):
        super(TargInvTrans, self).__init__(another._obj)

    #def fit(self, y):
    #    raise RuntimeError('TargInvTrans cannot be fitted')

    def _fit_trans_func(self, y):
        return self._trans_func(y)

    def _trans_func(self, y):
        return self._obj.inverse_transform(y)

class Clf(SkComp):

    def __init__(self, obj, proba=None):
        super(Clf, self).__init__(obj)
        self._cal_proba = proba == 'with' or proba == 'only'
        self._cal_class = proba != 'only'

    def fit(self, data):
        X, y = data
        self._obj.fit(X, y)

    def trans(self, ds):
        X, y = ds
        sy = cy = None
        if self._cal_proba:
            score = self._predict_proba(X)
            sy = dataset.dssy(score, y)
        if self._cal_class:
            cls = self._obj.predict(X)
            cy = dataset.dscy(cls, y)

        if cy is not None and sy is not None:
            return [cy, sy]
        elif sy is not None:
            return sy
        else:
            return cy

    def _predict_proba(self, X):
        if hasattr(self._obj, 'decision_function'):
            score = self._obj.decision_function(X)
        else:
            logger.info('no "decision_function" found, use "predict_proba" instead')
            score = self._obj.predict_proba(X)
            # if it is a binary classification problem, return a 1-D array
            if score.shape[1] == 2:
                score = score[:,1]
        #print '>>>>>', score
        #print '>>>>>', self._obj.classes_
        #sys.exit()
        # of probablities of class 1
        #    #print score
        #sys.exit()
        return score

class Dummy(Clf):

    def trans(self, ds):
        X, y = ds
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return Clf.trans(self, (X, y))

class Eva(mixin.EvaMixin, SkComp):

    def __init__(self, obj):
        func, args, opt = obj
        super(Eva, self).__init__(func)
        self.args = args
        self.opt = opt

    def _cal_func(self, y, score):
        return self._obj(y, score, *self.args, **self.opt)


def main():
    return

if __name__ == '__main__':
    main()
