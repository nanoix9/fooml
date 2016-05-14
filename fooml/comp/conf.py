#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from . import sk
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


def main():
    #obj = create_classifier('NB')
    return

if __name__ == '__main__':
    main()
