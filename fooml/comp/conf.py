#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from . import sk

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
    return ConfigEntry(comp_class, package + '.' + submodule, clazz)


__sklearn_config = {
        'LR': (sk.Sup, 'linear_model', 'LogisticRegression'),

        'AUC': (sk.Eva, 'metrics', 'roc_auc_score'),
        }

__config = {
        'sklearn': __sklearn_config,
        }


def main():
    #obj = create_classifier('NB')
    return

if __name__ == '__main__':
    main()
