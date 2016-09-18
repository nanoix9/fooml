#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import inspect
from fooml.log import logger
from fooml import util
from fooml import env
from fooml.proc import LazyObj


class Comp(object):

    def __init__(self, obj):
        self._obj = obj
        self._name = None

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def get_obj(self):
        if isinstance(self._obj, LazyObj):
            logger.info('create real object from LazyObj')
            self._obj = self._obj.init()
        return self._obj

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

    def add_callback(self, every=None, before=None, after=None, \
            before_train=None, after_train=None, \
            before_test=None, after_test=None):
        self._add_callback('before_train', before_train)
        self._add_callback('before_train', before)
        self._add_callback('before_train', every)
        self._add_callback('after_train', after_train)
        self._add_callback('after_train', after)
        self._add_callback('after_train', every)
        self._add_callback('before_test', before_test)
        self._add_callback('before_test', before)
        self._add_callback('before_test', every)
        self._add_callback('after_test', after_test)
        self._add_callback('after_test', after)
        self._add_callback('after_test', every)
        return self

    def _add_callback(self, name, callbacks):
        if not callbacks:
            return self
        var_name = '_callback_' + name
        if not hasattr(self, var_name):
            setattr(self, var_name, [])
        getattr(self, var_name).append(callbacks)
        return self

    def before_train(self, data):
        return self._callback_if_exist('_callback_before_train', data)

    def after_train(self, data, out):
        return self._callback_if_exist('_callback_after_train', data, out)

    def before_test(self, data):
        return self._callback_if_exist('_callback_before_test', data)

    def after_test(self, data, out):
        return self._callback_if_exist('_callback_after_test', data, out)

    def _callback_if_exist(self, callback_name, *args, **kwds):
        if hasattr(self, callback_name):
            for cb in getattr(self, callback_name):
                logger.info('invoke callback[%s]: "%s"' % (callback_name, cb.__name__))
                cb(*args, **kwds)

    def set_env(self, key, val, when='every'):
        cb = lambda *args: env.set_env(key, val, *args)
        cb.__name__ = 'set_env'
        kwds = {when: cb}
        self.add_callback(**kwds)
        return self

    def _extr_desc(self):
        return ''

class Nop(Comp):

    def __init__(self):
        super(Nop, self).__init__(None)

    def __repr__(self):
        return util.get_type_fullname(self)

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

