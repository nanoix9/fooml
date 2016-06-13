#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
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

class TargTrans(SkComp):

    def __init__(self, obj):
        super(TargTrans, self).__init__(obj)

    def fit_trans(self, data):
        return self._exec(data, self._obj.fit_transform)

    def trans(self, data):
        return self._exec(data, self._obj.transform)

    def _exec(self, data, func):
        if isinstance(data, dataset.dsxy):
            return self.__exec_xy(data, func)
        elif isinstance(data, dataset.dstv):
            pass
        else:
            raise TypeError()

    def _exec_xy(self, data, func):
        X, y = data
        if y is None:
            out = None
        else:
            out = func(y)
        dtran = dataset.dsxy(X, y)
        return dtran

class TargInvTrans(TargTrans):

    def __init__(self, another):
        super(TargInvTrans, self).__init__(another._obj)

    def fit(self, data):
        raise RuntimeError('TargInvTrans cannot be fitted')

    def fit_trans(self, data):
        return self.trans(data)

    def trans(self, data):
        return self._exec(data, self._obj.inverse_transform)

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
