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
    else:
        nrows = None
    foo.load_csv('ds_num', train_path='train_numeric.csv', target='Response', opt=dict(index_col='Id', nrows=nrows))
    #foo.load_csv('ds_date', train_path='train_date.csv', opt=dict(index_col='Id', nrows=nrows))

    pproc = fooml.feat_map('fillna', lambda x: x.fillna(0))
    #pproc = fooml.nop()
    xgbr = fooml.classifier('xgbr', 'xgboost', proba='only', opt=dict(params=dict()))
    rf = fooml.classifier('rf', 'randomforest', proba='only', opt=dict())
    auc = fooml.new_comp('auc', 'auc')

    foo.add_comp(pproc, 'ds_num', 'ds_filled')
    #foo.add_comp(xgbr, 'ds_num', 'proba')
    foo.add_comp(rf, 'ds_filled', 'proba')
    foo.add_comp(auc, 'proba')

    foo.compile()
    foo.run_train()
    return

if __name__ == '__main__':
    main()
