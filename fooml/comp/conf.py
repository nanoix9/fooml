#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from . import sk
from . import kr
from . import misc

DEFAULT = '__default__'
ANY = '__any__'

class ConfigEntry(object):

    def __init__(self, comp_class, module, clazz, arg=None, opt=None):
        self.comp_class = comp_class
        self.module = module
        self.clazz = clazz
        self.arg = arg
        self.opt = opt

def get_config(package, name):
    if package == ANY:
        for p, c in __config.iteritems():
            #print p, c
            if name in c:
                conf = c[name]
                package = p
                break
        else:
            raise ValueError('cannot find %s' % name)
    else:
        conf = __config[package][name]
    comp_class = conf[0]
    submodule = conf[1]
    clazz = conf[2]
    arg = None if len(conf) < 4 else conf[3]
    opt = None if len(conf) < 5 else conf[4]

    if package == DEFAULT:
        full_module = submodule
    else:
        full_module = package + '.' + submodule
    return ConfigEntry(comp_class, full_module, clazz, arg, opt)

def get_comp_class(module, name):
    return __get_from_config(__comp_conf, module, name)

def get_inv_comp_class(module, name):
    return __get_from_config(__inv_comp_conf, module, name)

def instant(name):
    return not name in __no_instant_set

def __get_from_config(config, module, name):
    def _get_default(d, k):
        if k in d:
            return d[k]
        else:
            return d[DEFAULT]

    module_path = module.split('.')
    v = config
    for p in module_path:
        if not isinstance(v, dict):
            break
        v = _get_default(v, p)
    if isinstance(v, dict):
        v = _get_default(v, name)
    return v


__default_config = {
        'binclass': (misc.TargFuncMapComp, 'fooml.proc', 'binclass'),
        'decide': (misc.DecideComp, 'fooml.proc', 'decide'),
        #'dummy': (misc.MyComp, 'fooml.proc', 'Dummy'),
        'dummy': (misc.FeatObjMergeComp, 'fooml.proc', 'Dummy'),
        'align_index': (misc.FeatFuncMergeComp, 'fooml.proc', 'align_index'),
        'merge': (misc.FeatFuncMergeComp, 'fooml.proc', 'merge'),

        'vgg_preproc': (misc.FeatFuncMapComp, 'fooml.comp.special', 'vgg_preproc'),
        'vgg19': (kr.Clf, 'fooml.comp.special', 'vgg19'),
        'vgg16': (kr.Clf, 'fooml.comp.special', 'vgg16'),
        }

__sklearn_config = {
        'split': (misc.SplitComp, 'cross_validation', 'train_test_split'),
        'partsplit': (misc.PartSplitComp, 'cross_validation', 'train_test_split'),

        'targetencoder': (sk.TargMap, 'preprocessing', 'LabelEncoder'),

        'LR': (sk.Clf, 'linear_model', 'LogisticRegression'),
        'DecisionTree': (sk.Clf, 'tree', 'DecisionTreeClassifier'),
        #'random': (sk.Dummy, 'dummy', 'DummyClassifier', None, dict(strategy='stratified')),
        'random': (sk.Dummy, 'dummy', 'DummyClassifier', None, dict(strategy='uniform')),

        'AUC': (sk.Eva, 'metrics', 'roc_auc_score'),
        'logloss': (sk.Eva, 'metrics', 'log_loss'),
        'report': (sk.Eva, 'metrics', 'classification_report'),

        'kfold': (sk.CV, 'cross_validation', 'KFold'),
        'stratifiedkfold': (sk.CV, 'cross_validation', 'StratifiedKFold'),
        'labelkfold': (sk.CV, 'cross_validation', 'LabelKFold'),
        }

__keras_config = {
        #'logloss': (kr.Eva, 'metrics', 'log_loss'),
        'to_categorical': (misc.TargFuncMapComp, 'utils.np_utils', 'to_categorical'),
        }

__config = {
        'sklearn': __sklearn_config,
        'keras': __keras_config,
        DEFAULT: __default_config,
        }


__keras_comp_config = {
        DEFAULT: kr.KerasComp,
        'models': kr.Clf,
        }

__comp_conf = {
        'keras': __keras_comp_config
        }

__inv_comp_conf = {
        'fooml': {
            'comp': {
                'sk': {
                    'TargTrans': sk.TargInvTrans,
                    }
                }
            }
        }

__no_instant_set = set([
        'KFold',
        'StratifiedKFold',
        'LabelKFold'])


def main():
    #obj = create_classifier('NB')
    from keras.models import Sequential
    m = Sequential()
    print get_comp_class(m.__class__.__module__, m.__class__.__name__)
    return

if __name__ == '__main__':
    main()
