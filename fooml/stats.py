#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import dataset

def summary(data):
    if isinstance(data, dataset.dsxy):
        xdesc = _summary(data.X)
        ydesc = _summary(data.y)
        desc = ['summary of target y:',
                ydesc,
                'summary of feature X:',
                xdesc,
                ]
    elif isinstance(data, dataset.dssy):
        xdesc = _summary(data.score)
        ydesc = _summary(data.y)
        desc = ['summary of score:',
                xdesc,
                'summary of true value:',
                ydesc,
                ]
    elif isinstance(data, dataset.desc):
        desc = str(data)
    elif _is_test_data(data):
        desc = 'Testing data: ' + str(data)
    else:
        desc = _summary(data)
    return desc

def desc_cate(data):
    if isinstance(data, pd.DataFrame):
        s = CateDesc([(col, desc_cate_series(data[col])) for col in data])
    elif isinstance(data, pd.Series):
        name = data.name or 'data'
        s = pd.DataFrame(dict(name=desc_cate_series(data))).transpose()
    return s

def desc_cate_series(series, num=5):
    assert(isinstance(series, pd.Series))
    #series = pd.Categorical(series)
    cnt = series.size
    distinct = series.nunique()  # someone said len(unique()) is 3-15x faster
    s_all = pd.Series([cnt, distinct], index=['count', 'unique'])
    s = series.groupby(series).count()
    s.sort_values(inplace=True, ascending=False)
    s = s.head(num)
    return s_all.append(s)

def _summary(data):
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        raise ValueError('unknown data type: %s' % type(data))

    dh = df.head(5)
    dn = df.describe().transpose()
    dc = desc_cate(df)
    return ['size: %s' % str(df.shape), \
            'head n:', str(dh), \
            'take as numeric type:', str(dn), \
            'take as category type:', str(dc)]

def _is_test_data(data):
    ret = isinstance(data, (int, float, basestring)) \
            or (isinstance(data, (list, tuple, dict, set)) and len(data) < 10)
    return ret

class CateDesc(object):
    def __init__(self, d):
        self._data = d

    def __repr__(self):
        return __str__

    def __str__(self):
        str_list = [ str(pd.DataFrame({n:s}).transpose()) for n, s in self._data ]
        return '\n'.join(str_list)

def main():
    return

if __name__ == '__main__':
    main()
