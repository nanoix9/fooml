#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from scipy.sparse import csr_matrix, hstack
from fooml import util


def binclass(y, pos=lambda x: x > 0):
    return np.array([1 if pos(x) else 0 for x in y])
    #return np.apply_along_axis(pos, 0, y)

def decide(y, thresh=0.0):
    return np.array([1 if x > thresh else 0 for x in y])

def align_index(df, df_base):
    level = 0
    idx = df_base.index
    if isinstance(idx, pd.MultiIndex):
        return df.reindex(idx)
    else:
        #print df
        #print df_base
        #print idx
        return df.reindex(idx, level=level)

def merge(*x):
    if len(x) == 1:
        return x[0]
    if not all(isinstance(xi, type(x[0])) for xi in x):
        raise TypeError('all dataset for merging should be the same type')
    if isinstance(x[0], csr_matrix):
        return hstack(tuple(x), format='csr')


class Dummy(object):

    def __init__(self, key=None, cols=None, sparse=False, **kwds):
        #self._args = args
        self._key = key
        self._cols = cols
        self._sparse = sparse
        self._kwds = kwds
        self._le_dict = {}

    def trans(self, *x):
        return self._fit_or_trans(x, mode='trans')

    def fit_trans(self, *x):
        return self._fit_or_trans(x, mode='fit_trans')

    def _fit_or_trans(self, x, mode):
        if len(x) == 2:
            x_main = x[0]
            x_idx = x[1]
        else:
            x_main = x[0]
            x_idx = x[0]

        if isinstance(x_main, pd.DataFrame) and isinstance(x_idx, pd.DataFrame):
            return self._fit_or_trans_df(x_main, x_idx, mode=mode)
        elif isinstance(x_main, np.ndarray):
            return self._fit_or_trans_np(x_main, x_idx, mode=mode)
        else:
            raise TypeError()

    def _fit_or_trans_df(self, df_main, df_idx, mode):
        row_idx = self._get_row_index(df_main, df_idx)
        not_na_idx = row_idx.notnull()
        #print not_na_idx
        #print df_main
        row_idx = row_idx[not_na_idx]
        df_main = df_main[not_na_idx]
        #print df_main
        dummy_vars = []
        for col in self._cols:
            col_values = util.get_index_or_col(df_main, col)
            le = self._le_dict.setdefault(col, skpp.LabelEncoder())
            if mode == 'fit_trans':
                label_idx = le.fit_transform(col_values)
            elif mode == 'fit':
                label_idx = le.fit(col_values)
            elif mode == 'trans':
                label_idx = le.transform(col_values)
            #print df_main.shape, row_idx.shape, label_idx.shape
            if self._sparse == 'csr':
                dv = csr_matrix((np.ones(df_main.shape[0]), (row_idx, label_idx)))
            dummy_vars.append(dv)
        #print dummy_vars
        return self._merge_all(dummy_vars)

    def _merge_all(self, dummy_vars):
        return merge(*dummy_vars)

    def _get_row_index(self, df, df_idx):
        if self._key is None or isinstance(self._key, basestring):
            key_idx = util.get_index_uniq_values(df_idx, self._key)
            #print key_idx
            df_idx_tmp = pd.DataFrame(dict(__ROWINDEX__=np.arange(len(key_idx))), index=key_idx)
            #print df_idx_tmp
        else:
            df_idx_tmp = self._key(df_idx)

        dftmp = df.merge(df_idx_tmp, how='left', left_index=True, right_index=True)
        #print dftmp
        row_idx = dftmp['__ROWINDEX__']
        #print row_idx
        #print row_idx
        return row_idx


def test_binclass():
    a = np.arange(5)
    print a
    print binclass(a)

def main():
    test_binclass()
    return

if __name__ == '__main__':
    main()
