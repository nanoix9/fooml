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
        row_idx, nrows_out = self._get_row_index(df_main, df_idx)
        not_na_idx = row_idx.notnull()
        #print not_na_idx
        #print df_main
        row_idx = row_idx[not_na_idx]
        df_main = df_main[not_na_idx]
        #print df_main
        if self._cols is None:
            cols = df_main.columns
        else:
            cols = self._cols
        dummy_vars = []
        for col in cols:
            #print df_main, col
            col_values = util.get_index_or_col(df_main, col)
            le = self._le_dict.setdefault(col, skpp.LabelEncoder())
            if mode == 'fit_trans':
                label_idx = le.fit_transform(col_values)
                nrows_seen = df_main.shape[0]
                row_seen_idx = row_idx
            elif mode == 'fit':
                label_idx = le.fit(col_values)
            elif mode == 'trans':
                seen_idx = np.in1d(col_values, le.classes_)
                label_idx = le.transform(col_values[seen_idx])
                nrows_seen = np.sum(seen_idx)
                row_seen_idx = row_idx[seen_idx]
            #print df_main.shape, row_idx.shape, label_idx.shape
            if self._sparse == 'csr':
                #dv = csr_matrix((np.ones(df_main.shape[0]), (row_idx, label_idx)))
                #print row_idx
                dv = csr_matrix((np.ones(nrows_seen), (row_seen_idx, label_idx)), \
                        shape=(nrows_out, le.classes_.shape[0]))
            dummy_vars.append(dv)
        #print dummy_vars
        return self._merge_all(dummy_vars), df_idx.index

    def _merge_all(self, dummy_vars):
        return merge(*dummy_vars)

    def _get_row_index(self, df, df_idx):
        if self._key is None or isinstance(self._key, basestring):
            key_idx = util.get_index_uniq_values(df_idx, self._key)
            #print key_idx
            nrows = len(key_idx)
            #print df_idx_tmp
            idx_array = np.arange(nrows)
        else:
            idx_array = self._key(df_idx)
            nrows = np.max(idx_array) + 1

        df_idx_tmp = pd.DataFrame(dict(__ROWINDEX__=idx_array), index=key_idx)
        if isinstance(self._key, basestring) and self._key in df.columns.values:
            dftmp = df[[self._key]].merge(df_idx_tmp, how='left', left_on=self._key, right_index=True)
        else:
            dftmp = df[[]].merge(df_idx_tmp, how='left', left_index=True, right_index=True)
        #print dftmp
        row_idx = dftmp['__ROWINDEX__']
        #print row_idx
        #print row_idx
        return row_idx, nrows


class LazyObj(object):

    def __init__(self, init):
        self._init = init

    def init(self, *args, **kwds):
        return self._init(*args, **kwds)

#class LazyObj(object):
#
#    def __init__(self, init):
#        self._init = init
#        self._obj = None
#        self._in_func = in_func
#        setattr(self, in_func, self.init_and_call)

#    def init_and_call(self, *args, **kwds):
#        print args
#        print kwds
#        if self._obj is None:
#            self._obj = self._init(*args, **kwds)
#        return getattr(self._obj, self._in_func)(*args, **kwds)

#    def is_init(self):
#        return self._obj is not None


def test_binclass():
    a = np.arange(5)
    print a
    print binclass(a)

def test_dummy():
    dmy = Dummy(sparse='csr')
    x = pd.DataFrame(dict(x=list('abcdeae')))
    print x
    y = dmy.fit_trans(x)
    print y.shape, y
    x2 = pd.DataFrame(dict(x=list('abxyb')))
    print x2
    y2 = dmy.trans(x2)
    print y2.shape, y2

    dmy = Dummy(key='idx', cols=['label'], sparse='csr')
    index = pd.MultiIndex.from_product([[10, 20], ['foo', 'bar', 'fb']], names=['idx', 'label'])
    x3 = pd.DataFrame(dict(x=list('abcdad')), index=index)
    index2 = pd.MultiIndex.from_product([[10, 20], ['foo', 'bar', 'foobar']], names=['idx', 'label'])
    x4 = pd.DataFrame(dict(x=list('abcebe')), index=index2)
    print x3
    y3 = dmy.fit_trans(x3)
    print y3
    y4 = dmy.trans(x4)
    print y4

def main():
    #test_binclass()
    test_dummy()
    return

if __name__ == '__main__':
    main()
