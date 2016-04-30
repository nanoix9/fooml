#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import comp.sk as skcomp
import sklearn
import importlib

class __ConfigEntry(object):

    def __init__(self, comp_class, module, clazz):
        self.comp_class = comp_class
        self.module = module
        self.clazz = clazz

def create_classifier(name, package='sklearn', args=[], opt={}):
    try:
        conf = __get_config(package, name)
        obj = create_obj(conf.module, conf.clazz, args, opt)
    except KeyError:
        obj = create_obj(package, name, args, opt)
    acomp = conf.comp_class(obj)
    return acomp

def create_obj(package, name, args=[], opt={}):
    #obj = create_or_default(package, name, args, opt)
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
    obj = clazz(*args, **opt)
    return obj

def __get_config(package, name):
    conf = __config[package][name]
    comp_class = conf[0]
    submodule = conf[1]
    clazz = conf[2]
    return __ConfigEntry(comp_class, package + '.' + submodule, clazz)


__sklearn_config = {
        'LR': (skcomp.Sup, 'linear_model', 'LogisticRegression'),
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
