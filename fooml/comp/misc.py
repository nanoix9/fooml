#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from comp import StatelessComp
import mixin
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger


class FunComp(StatelessComp):

    def __init__(self, fun_with_arg):
        if isinstance(fun_with_arg, tuple):
            fun, args, opt = fun_with_arg[0], [], {}
            if len(fun_with_arg) > 1:
                args = fun_with_arg[1]
            if len(fun_with_arg) > 2:
                opt = fun_with_arg[2]
        else:
            fun = fun_with_arg
        super(FunComp, self).__init__(fun)
        self._args = args
        self._opt = opt

    def __repr__(self):
        full_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return '%s(\n  func=%s,\n  args=%s,\n  opt=%s)' \
                % (full_name, (str(self._obj.__name__)), self._args, self._opt)

    def trans(self, data):
        return self._obj(data, *self._args, **self._opt)

    def _log_func(self):
        logger.info('call function "%s(%s, %s)"' % \
                (self._obj.__name__, \
                ', '.join(str(a) for a in self._args), \
                ', '.join('{}={}'.format(k, v) for k, v in self._opt.iteritems())))



class DsTransComp(FunComp):

    def __init__(self, fun_with_arg, on=['X', 'y']):
        super(DsTransComp, self).__init__(fun_with_arg)
        self.on = on

    def fit_trans(self, data):
        return self.trans(data)

    def trans(self, data):
        if not isinstance(data, dataset.dataset):
            return data

        def _foo(name, value):
            if value is None:
                out = None
            elif self._is_exec(name):
                out = self._call_func(value)
            else:
                out = value
            return out
        dtran = dataset.map(_foo, data)
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


class FuncTransComp(FunComp):

    def __init__(self, fun_with_arg):
        super(FuncTransComp, self).__init__(fun_with_arg)
        self._fit_func = None

    def _trans_func(self, data):
        self._log_func()
        return self._obj(data, *self._args, **self._opt)

    def _fit_trans_func(self, data):
        return self._trans_func(data)

class TargTransComp(mixin.TargTransMixin, FuncTransComp):
    pass

class FeatTransComp(mixin.FeatTransMixin, FuncTransComp):
    pass

class SplitComp(mixin.SplitMixin, FunComp):

    def __init__(self, fun_with_arg):
        super(SplitComp, self).__init__(fun_with_arg)
        self._fit_func = None

    def _split(self, X, y, index):
        self._log_func()
        ret = self._obj(X, y, *self._args, **self._opt)
        #TODO: not support index yet
        return ret + [None, None]

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
        return dataset.dscy(cls, data.y, data.index)


def main():
    return

if __name__ == '__main__':
    main()
