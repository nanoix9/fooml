#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger


class DsMixin(object):

    def get_train_valid(self, data):
        if isinstance(data, dataset.dsxy):
            X_train, y_train = data
            X_valid, y_valid = None, None
        elif isinstance(data, dataset.dstv):
            (X_train, y_train), ds_valid = data
            if ds_valid is None:
                X_valid, y_valid = None, None
            else:
                X_valid, y_valid = ds_valid
        else:
            raise TypeError('Unknown dataset type: %s' % data.__class__)
        return X_train, y_train, X_valid, y_valid

class BaseMixin(object):

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

class TransMixin(BaseMixin):

    def fit(self, data):
        return self._apply(data, self._fit_func)

    def fit_trans(self, data):
        return self._apply(data, self._fit_trans_func)

    def trans(self, data):
        return self._apply(data, self._trans_func)

class TargTransMixin(TransMixin):

    def _apply(self, data, func):
        return dataset.mapy(func, data)

class FeatTransMixin(TransMixin):

    def _apply(self, data, func):
        return dataset.mapx(func, data)


#class ClassifierMixin(BaseMixin):
#
#    #def fit(self, data):
#
#    def trans(self, ds):

class EvaMixin(BaseMixin):

    def fit(self, data):
        pass

    def trans(self, data):
        eva_list = []
        for d in slist.iter_multi(data, strict=True):
            if not isinstance(d, (dataset.dssy, dataset.dscy)):
                raise TypeError('data must be dssy or dscy type')
            score, y = d
            if y is None:
                eva = np.nan
            else:
                eva = self._cal_func(y, score)
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
