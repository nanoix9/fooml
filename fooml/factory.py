#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import comp.conf
import comp.special
import importlib
import inspect
import util
from log import logger

class FakeObj(object):

    def __str__(self):
        return 'FakeObj(None)'

_fake_obj = FakeObj()

def create_inv_trans(acomp):
    assert(acomp is not None)
    comp_class = comp.conf.get_inv_comp_class(acomp.__class__.__module__, acomp.__class__.__name__)
    logger.info('create inverse component "%s.%s"' % \
            (comp_class.__module__, comp_class.__name__))
    return comp_class(acomp)

def obj2comp(obj, comp_opt={}):
    assert(obj is not None)
    comp_class = comp.conf.get_comp_class(obj.__class__.__module__, obj.__class__.__name__)
    if comp_class is None:
        return comp.Comp(obj, **comp_opt)
    else:
        return comp_class(obj, **comp_opt)

def create_comp(name, package=None, args=[], opt={}, comp_opt={}):
    if package is None:
        package = comp.conf.ANY

    try:
        conf = comp.conf.get_config(package, name)
        opt = util.merge_dict_or_none(opt, conf.opt)
        obj = create_obj(conf.module, conf.clazz, args, opt)
    except KeyError:
        obj = create_obj(package, name, args, opt)
    logger.info('create component "%s.%s" with "opt=%s"' % \
            (conf.comp_class.__module__, conf.comp_class.__name__, comp_opt))
    acomp = conf.comp_class(obj, **comp_opt)
    return acomp

def create_obj(package, name, args=[], opt={}):
    #obj = create_or_default(package, name, args, opt)
    logger.info('create object "%s(%s)" with "args=%s, opt=%s"' % \
            (name, package, str(args), str(opt)))
    if package == 'fooml.comp.special':
        obj = getattr(comp.special, name)(*args, **opt)
    elif name == 'None':
        obj = _fake_obj, args, opt
    else:
        obj = create_from_str(package, name, args, opt)
    return obj

#def create_or_default(package, name, args, opt):
#    try:
#        obj = create_from_str()
#    except KeyError:
#        #raise
#        obj = create_from_str(package, name, args, opt)
#    return obj

def create_from_str(module_name, clazz_name, args, opt):
    #print module_name
    try:
        module = importlib.import_module(module_name)
    except:
        logger.error('fail to load module "%s"' % (module_name))
        raise

    try:
        clazz = getattr(module, clazz_name)
    except:
        logger.error('no class "%s" in module "%s"' % (clazz_name, module_name))
        raise

    if inspect.isclass(clazz) and comp.conf.instant(clazz_name):
        obj = clazz(*args, **opt)
    elif hasattr(clazz, '__call__'):
        obj = clazz, args, opt
    return obj


def main():
    #obj = create_classifier('NB')
    obj = create_classifier('LR')
    print obj
    obj = create_obj('sklearn.svm', 'SVC')
    print obj
    return

if __name__ == '__main__':
    main()
