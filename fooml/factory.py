#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import comp.conf
import importlib
import inspect
import util
from log import logger


def create_classifier(name, package='sklearn', args=[], opt={}, comp_opt={}):
    return create_comp(package, name, args, opt, comp_opt)

def create_evaluator(name, package='sklearn', args=[], opt={}, comp_opt={}):
    return create_comp(package, name, args, opt, comp_opt)

def create_trans(name, package=comp.conf.ANY, args=[], opt={}, comp_opt={}):
    return create_comp(package, name, args, opt, comp_opt)

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

def create_comp(package, name, args, opt, comp_opt):
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
    obj = create_from_str(package, name, args, opt)
    return obj

def create_or_default(package, name, args, opt):
    try:
        obj = create_from_str()
    except KeyError:
        #raise
        obj = create_from_str(package, name, args, opt)
    return obj

def create_from_str(module_name, clazz_name, args, opt):
    module = importlib.import_module(module_name)
    try:
        clazz = getattr(module, clazz_name)
    except:
        logger.error('no class "%s" in module "%s"' % (clazz_name, module_name))
        raise
    if inspect.isclass(clazz):
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
