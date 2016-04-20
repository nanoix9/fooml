#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import sklearn
import importlib

class __ConfigEntry(object):

    def __init__(self, module, clazz):
        self.module = module
        self.clazz = clazz

def create_classifier(name, package='sklearn', args=[], opt={}):
    obj = create_obj(package, name, args, opt)
    acomp = comp.SklearnComp(obj)
    return acomp

def create_obj(package, name, args=[], opt={}):
    obj = create_or_default(package, name, args, opt)
    return obj

def create_or_default(package, name, args, opt):
    try:
        conf = __get_config(package, name)
        obj = create_from_str(conf.module, conf.clazz, args, opt)
    except KeyError:
        #raise
        obj = create_from_str(package, name, args, opt)
    return obj

def create_from_str(module_name, clazz_name, args, opt):
    module = importlib.import_module(module_name)
    clazz = getattr(module, clazz_name)
    obj = clazz(*args, **opt)
    return obj

def __get_config(package, name):
    conf = __config[package][name]
    submodule = conf[0]
    clazz = conf[1]
    return __ConfigEntry(package + '.' + submodule, clazz)


__sklearn_config = {
        'LR': ('linear_model', 'LogisticRegression'),
        }

__config = {
        'sklearn': __sklearn_config,
        }


def main():
    #obj = create_classifier('NB')
    obj = create_classifier('LR')
    print obj
    obj = create_obj('sklearn.svm', 'SVC')
    print obj
    return

if __name__ == '__main__':
    main()
