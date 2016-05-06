#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import comp.sk as skcomp
import comp.conf
import importlib
import inspect
from log import logger


def create_classifier(name, package='sklearn', args=[], opt={}):
    return create_comp(package, name, args, opt)

def create_evaluator(name, package='sklearn', args=[], opt={}):
    return create_comp(package, name, args, opt)

def create_comp(package, name, args, opt):
    try:
        conf = comp.conf.get_config(package, name)
        obj = create_obj(conf.module, conf.clazz, args, opt)
    except KeyError:
        obj = create_obj(package, name, args, opt)
    acomp = conf.comp_class(obj)
    return acomp

def create_obj(package, name, args=[], opt={}):
    #obj = create_or_default(package, name, args, opt)
    logger.info('create component "%s(%s)" with "args=%s, opt=%s"' % \
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
    clazz = getattr(module, clazz_name)
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
