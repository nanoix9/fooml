#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import comp
import mixin
from fooml import dataset
from fooml.log import logger
try:
    import xgboost as xgb
except ImportError:
    pass


class XgboostComp(mixin.ClfMixin, comp.Comp):

    def __init__(self, obj, proba=None, params={}, nrounds=1000):
        super(XgboostComp, self).__init__(obj)
        self.set_proba(proba)
        self._params = params
        self._nrounds = nrounds
        self._best_rounds = []

    def fit(self, data):
        if isinstance(data, dataset.dsxy):
            Xt, yt = data
            Xv, yv = None, None
            logger.info('best number of rounds: %s' % self._best_rounds)
            nrounds = self._best_rounds[-1] + 1
        elif isinstance(data, dataset.dstv):
            Xt, yt = data.train
            Xv, yv = data.valid
            nrounds = self._nrounds
        else:
            raise TypeError()
        nb_class = len(np.unique(yt))
        params = self.__get_default_params(nb_class)
        params.update(self._params)
        params['num_class'] = len(np.unique(yt))

        d_train = xgb.DMatrix(Xt, label=yt)
        watchlist = [(d_train, 'train')]
        kwds = {}
        if Xv is not None:
            d_valid = xgb.DMatrix(Xv, label=yv)
            watchlist.append((d_valid, 'valid'))
            kwds['early_stopping_rounds'] = 25
        self._obj = xgb.train(params, d_train, nrounds, watchlist, **kwds)
        if Xv is not None:
            self._best_rounds.append(self._obj.best_iteration)
        return self

    def _predict_proba(self, X):
        ntree_limit = 0
        if hasattr(self._obj, 'best_ntree_limit'):
            ntree_limit = self._obj.best_ntree_limit
            logger.info('xgbooster has best_ntree_limit=%d, use it for prediction' % ntree_limit)
        pred = self._obj.predict(xgb.DMatrix(X), ntree_limit=ntree_limit)
        #print pred
        return pred

    def __get_default_params(self, nb_class):
        if nb_class == 2:
            objective = 'binary:logistic'
        elif nb_class > 2:
            objective = 'multi:softprob'
        else:
            raise ValueError('invalid number of classes')
        xgb_params = {
                #'booster': 'gblinear',
                'booster': 'gbtree',
                'objective' : objective,
                'eval_metric' : 'mlogloss',
                'eta' : 0.005,
                'lambda' : 3,
                'alpha' : 2,
                }
        return xgb_params


def main():
    return

if __name__ == '__main__':
    main()
