#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import inspect
from fooml.log import logger


#class Component(object):
#
#    def __init__(self, name, obj):
#        self.name = name
#        self.obj = obj
#
#class _CompList(Component):
#
#    def __init__(self, name):
#        super(_CompList, self).__init__(name, [])
#
#    def add_obj(self, name, obj):
#        self.obj.append(Component(name, obj))
#
#    def add_component(self, name, comp):
#        self.obj.append(comp)
#
#    def __iter__(self):
#        for o in self.obj:
#            yield o
#
#class Parallel(_CompList):
#
#    def __init__(self, name='parallel'):
#        super(Parallel, self).__init__(name)
#        #self._objs = (name, [])
#
#class Serial(_CompList):
#
#    def __init__(self, name='serial'):
#        super(Serial, self).__init__(name)
#        #self._objs = []
#

class Comp(object):

    def __init__(self, obj):
        self._obj = obj

    def __str__(self):
        return repr(self)

    def __repr__(self):
        full_name = self.__class__.__module__ + '.' + self.__class__.__name__
        if type(self._obj) == type(lambda: 0):
            desc = '%s(\n  func=%s)' % (full_name, (str(self._obj.__name__)))
        else:
            desc = '%s(\n  obj=%s)' % (full_name, (str(self._obj)))
        return desc
        #return '%s(obj=%s)' % (self.__class__.__name__, self._obj)

    def fit(self, data):
        raise NotImplementedError()

    def trans(self, data):
        raise NotImplementedError()

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

class LambdaComp(Comp):

    def __init__(self, obj, fit, trans, fit_trans=None):
        super(LambdaComp, self).__init__(obj)
        self._fit = fit
        self._trans = trans
        if fit_trans is not None:
            self._fit_trans = fit_trans
        else:
            self._fit_trans = self._default_fit_trans

    def fit(self, data):
        if self._fit is not None:
            return self._fit(self._obj, data)
        return None

    def trans(self, data):
        logger.debug('trans of comp lambda: %s %s' % (self._obj, data))
        #print self._trans(self._obj, data)
        return self._trans(self._obj, data)

    def fit_trans(self, data):
        print '>>>> fit_trans of lambda comp:', self._obj, data
        return self._fit_trans(self._obj, data)

    def _default_fit_trans(self, _, data):
        self.fit(data)
        return self.trans(data)

class StatelessComp(Comp):

    def __init__(self, obj):
        super(StatelessComp, self).__init__(obj)

    def fit(self, data):
        return None

    def fit_trans(self, data):
        #print '>>>> fit_trans of stateless comp:', self._obj, data
        return self.trans(data)

class PassComp(StatelessComp):

    def __init__(self):
        super(PassComp, self).__init__('fake_obj')

    def trans(self, data):
        logger.debug('trans of comp pass through: %s, %s' % (self._obj, data))
        return data

    def fit_trans(self, data):
        logger.debug('fit_trans of comp pass through: %s, %s' % (self._obj, data))
        return data

class ConstComp(StatelessComp):

    def __init__(self, const=None):
        super(ConstComp, self).__init__(const)

    def trans(self, data):
        logger.debug('trans of comp const: %s, %s' % (self._obj, data))
        return self._obj

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

def main():
    return

if __name__ == '__main__':
    main()

