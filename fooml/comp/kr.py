#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger

class KerasComp(comp.Comp):

    def __init__(self, obj, train_opt={}):
        super(KerasComp, self).__init__(obj)
        self._train_opt = train_opt

    def fit(self, data):
        (X_train, y_train), ds_valid = data
        opt = dict(self._train_opt)
        logger.info('trainning nerual network with options: %s' % str(opt))
        if ds_valid is not None:
            opt['validation_data'] = (ds_valid.X, ds_valid.y)
            logger.info('and validation set: X%s, y%s' % (ds_valid.X.shape, ds_valid.y.shape))
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

def main():
    return

if __name__ == '__main__':
    main()
