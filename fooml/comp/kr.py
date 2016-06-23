#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import mixin
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger

class KerasComp(comp.Comp):
    pass

class Clf(KerasComp):

    def __init__(self, obj, train_opt={}):
        super(KerasComp, self).__init__(obj)
        self._train_opt = train_opt

    def fit(self, data):
        X_train, y_train, X_valid, y_valid = dataset.get_train_valid(data)
        opt = dict(self._train_opt)
        logger.info('trainning nerual network with options: %s' % str(opt))
        if X_valid is not None:
            opt['validation_data'] = (X_valid, y_valid)
            logger.info('and validation set: X%s, y%s' % (X_valid.shape, y_valid.shape))
        else:
            logger.info('and no validation set')
        return self._obj.fit(X_train, y_train, **opt)

    def trans(self, data):
        X, y = data
        return dataset.dssy(self._obj.predict(X), y)

    def fit_trans(self, data):
        if isinstance(data, dataset.dstv):
            ds_train, ds_valid = data
        else:
            ds_train = data
        self.fit(data)
        return self.trans(ds_train)

        #print X, y
        #self._obj.fit_transform(X, y)
        #print self._obj
        #return self._obj.fit_transform(X, y)

class Eva(mixin.EvaMixin, KerasComp):

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
