#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger


class BaseMixin(object):

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

class TargTransMixin(BaseMixin):

    def fit(self, data):
        return self._apply(data, self._fit_func)

    def fit_trans(self, data):
        return self._apply(data, self._fit_trans_func)

    def trans(self, data):
        return self._apply(data, self._trans_func)

    def _apply(self, data, func):
        logger.info('call function "%s"' % func.__name__)
        if isinstance(data, dataset.dsxy):
            return self._apply_xy(data, func)
        elif isinstance(data, dataset.dscy):
            return self._apply_cy(data, func)
        elif isinstance(data, dataset.dstv):
            pass
        else:
            raise TypeError('not supported data type: %s' % data.__class__)

    def _apply_xy(self, data, func):
        X, y = data
        if y is None:
            out = None
        else:
            out = func(y)
        dtran = dataset.dsxy(X, out)
        return dtran

    def _apply_cy(self, data, func):
        c, y = data
        if y is None:
            yout = None
        else:
            yout = func(y)
        cout = func(c)
        dtran = dataset.dscy(cout, yout)
        return dtran

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
