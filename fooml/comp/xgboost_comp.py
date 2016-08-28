#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import mixin
try:
    import xgboost as xgb
except ImportError:
    pass


class XgboostComp(mixin.ClfMixin, comp.Comp):

    def __init__(self, obj, proba=None):
        super(XgboostComp, self).__init__(obj)
        self.set_proba(proba)

    def fit(self, data):
        Xt, yt = data.train
        Xv, yv = data.valid
        params = {}
        params['booster'] = 'gblinear'
        params['objective'] = "multi:softprob"
        params['eval_metric'] = 'mlogloss'
        params['eta'] = 0.005
        params['num_class'] = 12
        params['num_class'] = 2
        params['lambda'] = 3
        params['alpha'] = 2

        d_train = xgb.DMatrix(Xt, label=yt)
        d_valid = xgb.DMatrix(Xv, label=yv)

        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        self._obj = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)

    def _predict_proba(self, X):
        pred = self._obj.predict(xgb.DMatrix(X))
        return pred


def main():
    return

if __name__ == '__main__':
    main()
