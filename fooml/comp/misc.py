#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import comp
import mixin
import group
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger


class MyComp(comp.Comp):

    def __init__(self, obj):
        super(MyComp, self).__init__(obj)

    def fit(self, *args, **kwds):
        return self._obj.fit(*args, **kwds)

    def fit_trans(self, *args, **kwds):
        return self._obj.fit_trans(*args, **kwds)

    def trans(self, *args, **kwds):
        return self._obj.trans(*args, **kwds)

class FunComp(comp.StatelessComp):

    def __init__(self, fun_with_arg):
        if isinstance(fun_with_arg, tuple):
            fun, args, opt = fun_with_arg[0], [], {}
            if len(fun_with_arg) > 1:
                args = fun_with_arg[1]
            if len(fun_with_arg) > 2:
                opt = fun_with_arg[2]
        else:
            fun = fun_with_arg
            args = []
            opt = {}
        super(FunComp, self).__init__(fun)
        self._args = tuple(args)
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


class FuncTransComp(FunComp):

    def __init__(self, fun_with_arg):
        super(FuncTransComp, self).__init__(fun_with_arg)
        self._fit_func = None

    def _trans_func(self, *data):
        self._log_func()
        #print data, self._args
        return self._obj(*(data + self._args), **self._opt)

    def _fit_trans_func(self, *data):
        return self._trans_func(*data)

class TargFuncMapComp(mixin.TargMapMixin, FuncTransComp):
    pass

class FeatFuncMapComp(mixin.FeatMapMixin, FuncTransComp):
    pass

class FeatFuncMergeComp(mixin.FeatMergeMixin, FuncTransComp):
    pass

class ObjTransComp(comp.Comp):

    def __init__(self, obj):
        super(ObjTransComp, self).__init__(obj)
        self._fit_func = None

    def _trans_func(self, *data):
        return self._obj.trans(*data)

    def _fit_trans_func(self, *data):
        return self._obj.fit_trans(*data)

class FeatObjMapComp(mixin.FeatMapMixin, ObjTransComp):
    pass

class FeatObjMergeComp(mixin.FeatMergeMixin, ObjTransComp):
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

class PartSplitComp(mixin.PartSplitMixin, FunComp):

    def __init__(self, fun_with_arg, part_key=lambda x:x, obj_return_value=True):
        super(PartSplitComp, self).__init__(fun_with_arg)
        self._fit_func = None
        self._part_key = part_key
        self._obj_return_value = obj_return_value

    def _get_labels(self, data, partition):
        return np.array(self._part_key(partition.X)), partition.get_index()

    def _split_labels_index(self, labels):
        keys = np.unique(labels)
        self._log_func()
        tk, vk = self._obj(keys, *self._args, **self._opt)
        if self._obj_return_value:
            tk = self._value_to_index(tk, labels)
            vk = self._value_to_index(vk, labels)
        return tk, vk


##################### Deprecated

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
