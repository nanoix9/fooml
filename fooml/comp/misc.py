#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from comp import FunComp
from fooml import dataset
from fooml.dt import slist
from fooml import util

class DsTransComp(FunComp):

    def __init__(self, fun_with_arg, on=['X', 'y']):
        super(DsTransComp, self).__init__(fun_with_arg)
        self.on = on

    def fit_trans(self, data):
        return self.trans(data)

    def trans(self, data):
        if not isinstance(data, dataset.dataset):
            return data

        dtran = clazz = data.__class__()
        for name, value in util.getmembers(data):
            #print '>>>', name, value
            if value is None:
                out = None
            elif self._is_exec(name):
                out = self._call_func(value)
            else:
                out = value
            setattr(dtran, name, out)
        return dtran

    #def fit_trans(self, data):
    #    X, y = data
    #    X_out, y_out = X, y
    #    if self._is_exec('X'):
    #        X_out = self._call_func(X)
    #    if self._is_exec('y'):
    #        y_out = self._call_func(y)
    #    return dataset.dsxy(X_out, y_out)

    def _is_exec(self, name):
        return slist.contains(self.on, name)

    def _call_func(self, data):
        return FunComp.trans(self, data)


class TargTransComp(DsTransComp):

    def __init__(self, fun_with_arg):
        super(TargTransComp, self).__init__(fun_with_arg, 'y')

class ScoreComp(DsTransComp):

    def __init__(self, fun_with_arg):
        super(ScoreComp, self).__init__(fun_with_arg, 'score')

class DecideComp(DsTransComp):

    def __init__(self, fun_with_arg):
        super(DecideComp, self).__init__(fun_with_arg, 'score')

    def trans(self, data):
        if not isinstance(data, dataset.dssy):
            raise ValueError('data should be fooml.dataset.dssy type')
        cls = self._call_func(data.score)
        return dataset.dscy(cls, data.y)


def main():
    return

if __name__ == '__main__':
    main()
