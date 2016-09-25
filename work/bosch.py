#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import fooml
from fooml import env
from fooml.log import logger


def main():
    #print sys.argv
    foo = fooml.FooML('bosch')
    #foo.parse_args(sys.argv[1:])
    foo.parse_args()

    foo.set_data_home('/vola1/scndof/data/bosch')
    #foo.enable_data_cache()

    if foo.debug:
        nrows = 1000
        nb_fold = 2
    else:
        nrows = None
        nb_fold = 5
    foo.load_csv('ds_num', train_path='train_numeric.csv', target='Response', opt=dict(index_col='Id', nrows=nrows))
    #foo.load_csv('ds_date', train_path='train_date.csv', opt=dict(index_col='Id', nrows=nrows))

    pproc = fooml.feat_map('fillna', lambda x: x.fillna(0))
    #pproc = fooml.nop()
    xgbr = fooml.classifier('xgbr', 'xgboost', proba='only', opt=dict(params=dict()))
    rf = fooml.classifier('rf', 'randomforest', proba='only', opt=dict())
    auc = fooml.new_comp('auc', 'auc')
    use_dstv = False

    foo.add_comp(pproc, 'ds_num', 'ds_filled')
    #foo.add_comp(xgbr, 'ds_num', 'proba')
    cv_clf = fooml.submodel('cv_clf', input='ds_filled', output=['y_proba', 'ds_auc'])
    cv_clf.add_comp(rf, 'ds_filled', 'proba')
    cv_clf.add_comp(auc, 'proba', 'ds_auc')

    cv = fooml.cross_validate('cv', cv_clf, eva='ds_auc', k=nb_fold, type='stratifiedkfold', use_dstv=use_dstv)
    foo.add_comp(cv, 'ds_filled', ['y_proba', 'ds_cv'])

    foo.compile()
    foo.run_train()
    return

if __name__ == '__main__':
    main()
