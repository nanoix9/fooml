#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from . import sk
from . import kr
from . import misc

DEFAULT = '__default__'

class ConfigEntry(object):

    def __init__(self, comp_class, module, clazz):
        self.comp_class = comp_class
        self.module = module
        self.clazz = clazz

def get_config(package, name):
    conf = __config[package][name]
    comp_class = conf[0]
    submodule = conf[1]
    clazz = conf[2]

    if package == DEFAULT:
        full_module = submodule
    else:
        full_module = package + '.' + submodule
    return ConfigEntry(comp_class, full_module, clazz)

def get_comp_class(module, name):
    def _get_default(d, k):
        if k in d:
            return d[k]
        else:
            return d[DEFAULT]

    module_path = module.split('.')
    v = __comp_conf
    for p in module_path:
        if not isinstance(v, dict):
            break
        v = _get_default(v, p)
    if isinstance(v, dict):
        v = _get_default(v, name)
    return v

__default_config = {
        'binclass': (misc.TargTransComp, 'fooml.proc', 'binclass'),
        'decide': (misc.DecideComp, 'fooml.proc', 'decide'),
        }

__sklearn_config = {
        'LR': (sk.Clf, 'linear_model', 'LogisticRegression'),
        'DecisionTree': (sk.Clf, 'tree', 'DecisionTreeClassifier'),

        'AUC': (sk.Eva, 'metrics', 'roc_auc_score'),
        'report': (sk.Eva, 'metrics', 'classification_report'),
        }

__config = {
        'sklearn': __sklearn_config,
        DEFAULT: __default_config,
        }


__keras_comp_config = {
        DEFAULT: kr.KerasComp
        }

__comp_conf = {
        'keras': __keras_comp_config
        }

def main():
    #obj = create_classifier('NB')
    from keras.models import Sequential
    m = Sequential()
    print get_comp_class(m.__class__.__module__, m.__class__.__name__)
    return

if __name__ == '__main__':
    main()
