#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import inspect
from fooml.log import logger
from fooml import util


class Comp(object):

    def __init__(self, obj):
        self._obj = obj
        self._name = None

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def __str__(self):
        return repr(self)

    def __repr__(self):
        full_name = util.get_type_fullname(self)
        if type(self._obj) == type(lambda: 0):
            desc = 'func=%s' % (str(self._obj.__name__))
        else:
            desc = 'obj=%s' % (str(self._obj))
        extr_desc = self._extr_desc()
        if extr_desc:
            desc = desc + '\n' + extr_desc
        desc = '%s(name="%s"\n%s)' % (full_name, self._name, util.indent(desc, 2))
        return desc
        #return '%s(obj=%s)' % (self.__class__.__name__, self._obj)

    def fit(self, data):
        raise NotImplementedError()

    def trans(self, data):
        raise NotImplementedError()

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

    def _extr_desc(self):
        return ''

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

def main():
    return

if __name__ == '__main__':
    main()

