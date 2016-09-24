#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from StringIO import StringIO
import numpy as np
import pandas as pd
import scipy.sparse as sp

import dataset
import util
import settings
from log import logger

__summary_cache = {}

settings.CACHE_DATA_SUMMARY_RESULT = False

def summary(data):
    #data_id = id(data)
    #if settings.CACHE_DATA_SUMMARY_RESULT:
    #    if data_id in __summary_cache:
    #        logger.debug('get data summary for 0x%x from cache' % data_id)
    #        return __summary_cache[data_id]

    if isinstance(data, dataset.dsxy):
        xdesc = summary(data.X)
        ydesc = summary(data.y)
        index = data.get_index()
        idesc = summary(index)
        if index is not None:
            idesc.extend(_check_unique(index))
        desc = ['type: %s' % util.get_type_fullname(data),
                'indices:',
                idesc,
                'summary of target y:',
                ydesc,
                'summary of feature X:',
                xdesc,
                ]
    elif isinstance(data, dataset.dssy):
        xdesc = summary(data.score)
        ydesc = summary(data.y)
        idesc = summary(data.index)
        desc = ['type: %s' % util.get_type_fullname(data),
                'indices:',
                idesc,
                'summary of score:',
                xdesc,
                'summary of true value:',
                ydesc,
                ]
    elif isinstance(data, dataset.dscy):
        xdesc = summary(data.cls)
        ydesc = summary(data.y)
        idesc = summary(data.index)
        desc = ['type: %s' % util.get_type_fullname(data),
                'indices:',
                idesc,
                'summary of predicted class:',
                xdesc,
                'summary of true class:',
                ydesc,
                ]
    elif isinstance(data, dataset.dstv):
        tdesc = summary(data.train)
        vdesc = summary(data.valid)
        desc = ['type: %s' % util.get_type_fullname(data),
                'summary of train set:',
                tdesc,
                'summary of validation set:',
                vdesc,
                ]
    elif isinstance(data, dataset.desc):
        desc = util.indent(str(data), ind=2)
    elif _is_small_data(data):
        desc = util.indent('value: ' + str(data), ind=2)
    else:
        desc = _summary(data)

    #if settings.CACHE_DATA_SUMMARY_RESULT:
    #    __summary_cache[data_id] = desc

    return desc

def _check_unique(arr):
    arr_uniq, counts = np.unique(arr, return_counts=True)
    if not np.all(counts == 1):
        ret = ['warning: NOT UNIQUE']
        non_uniq_idx = counts > 1
        df = pd.DataFrame(counts[non_uniq_idx], index=arr_uniq[non_uniq_idx], columns=['count'])
        df.index.name = 'value'
        ret.append(['most duplicated values and counts:', str(df.sort().head(5).tranpose())])
    else:
        ret = ['index is unique']
    return ret

def desc_cate(data):
    if isinstance(data, pd.DataFrame):
        d = []
        ncol = data.shape[1]
        for icol, col in enumerate(data):
            if ncol > 60:
                if icol == 30:
                    d.append(('...', '...'))
                    continue
                elif icol > 30 and icol < ncol - 30:
                    continue
            dc = desc_cate_series(data[col])
            if dc is not None:
                d.append((col, dc))
        if len(d) > 0:
            s = CateDesc(d)
        else:
            s = None
    elif isinstance(data, pd.Series):
        d = desc_cate_series(data)
        if isinstance(d, basestring):
            s = d
        elif isinstance(d, pd.Series):
            name = data.name or 'data'
            s = pd.DataFrame(dict(name=d)).transpose()
        else:
            raise RuntimeError()
            s = None
    return s

def desc_cate_series(series, num=5):
    assert(isinstance(series, pd.Series))
    #series = pd.Categorical(series)
    cnt = series.size
    distinct = series.nunique()  # someone said len(unique()) is 3-15x faster
    if distinct < cnt / 2:
        s_all = pd.Series([cnt, distinct], index=['count', 'unique'])
        s = series.groupby(series).count()
        s.sort_values(inplace=True, ascending=False)
        s = s.head(num)
        ret = s_all.append(s)
    else:
        ret = '  seems not to be categorical, %d(unique)/%d(total)' % (distinct, cnt)
    return ret

def _summary(data):
    if data is None:
        return '  data is NONE'

    ret = []
    if isinstance(data, pd.DataFrame):
        sio = StringIO()
        data.info(buf=sio, verbose=True)
        text = sio.getvalue().strip()
        arr = text.split('\n')
        if len(arr) > 60:
            lines = list(arr[0:30])
            lines.append('...')
            lines.extend(arr[-31:])
            text = '\n'.join(lines)
        ret.append('type info: %s' % util.indent(text, ind=2))
    else:
        ret.append('type: %s' % util.get_type_fullname(data))

    ret.append('size: %s' % str(data.shape))

    as_numeric = False
    if isinstance(data, (pd.Series, pd.DataFrame)):
        df = data
        if isinstance(data, pd.Series) or len(data.columns) > 0:
            as_numeric = True
    elif isinstance(data, (pd.MultiIndex)):
        df = pd.DataFrame(data)
    elif isinstance(data, (pd.Index)):
        df = pd.Series(data)
    elif isinstance(data, np.ndarray):
        dtype = data.dtype
        if dtype is not None and np.issubdtype(dtype, np.number):
            as_numeric = True
        if len(data.shape) == 1:
            df = pd.Series(data)
        elif len(data.shape) <= 2:
            df = pd.DataFrame(data)
        else:
            # TODO too large, skip it as workaround
            return ret
            #raise ValueError('data with more than 2 dimension is not supported yet')
            #df = pd.DataFrame(data.reshape(data.shape[0], -1))
            df = pd.DataFrame(data.flatten())
    elif isinstance(data, sp.csr_matrix):
        nz = data.count_nonzero()
        cnt = data.shape[0] * data.shape[1]
        ret.append('sparsity(element): %d(nonzero)/%d(total) = %f' % (nz, cnt, float(nz)/cnt))
        nz_row = np.sum(np.diff(data.indptr) != 0)
        nz_col = len(np.unique(data.indices))
        cnt_row, cnt_col = data.shape
        ret.append('sparsity(row):     %d(nonzero)/%d(total) = %f' % (nz_row, cnt_row, float(nz_row)/cnt_row))
        ret.append('sparsity(col):     %d(nonzero)/%d(total) = %f' % (nz_col, cnt_col, float(nz_col)/cnt_col))
        return ret
    else:
        raise ValueError('unknown data type: %s' % type(data))


    dh = df.head(5)
    dt = df.tail(5)
    ret.append('head n:')
    ret.append([str(dh)])
    ret.append('tail n:')
    ret.append([str(dt)])

    if as_numeric:
        dn = df.describe().transpose()
        ret.append('take as numeric type:')
        ret.append([str(dn)])

    dc = desc_cate(df)
    ret.append('take as category type:')
    ret.append([str(dc)])
    return ret

def _is_small_data(data):
    ret = isinstance(data, (int, float, basestring)) \
            or (isinstance(data, (tuple, dict, set)) and len(data) < 10)
    return ret

class CateDesc(object):
    def __init__(self, d):
        self._data = d

    def __repr__(self):
        return __str__

    def __str__(self):
        #print self._data
        str_list = [ '......' if n == '...' else \
                str(n) + '\n' + s if isinstance(s, basestring) \
                else str(pd.DataFrame({n:s}).transpose()) \
                for n, s in self._data ]
        return '\n'.join(str_list)

def main():
    return

if __name__ == '__main__':
    main()
