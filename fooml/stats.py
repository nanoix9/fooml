#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import dataset

def summary(data):
    if isinstance(data, dataset.dsxy):
        xdesc = summary(data.X)
        ydesc = summary(data.y)
        desc = ['summary of target y:',
                ydesc,
                'summary of feature X:',
                xdesc,
                ]
    elif isinstance(data, dataset.dssy):
        xdesc = summary(data.score)
        ydesc = summary(data.y)
        desc = ['summary of score:',
                xdesc,
                'summary of true value:',
                ydesc,
                ]
    elif isinstance(data, dataset.dscy):
        xdesc = summary(data.cls)
        ydesc = summary(data.y)
        desc = ['summary of predicted class:',
                xdesc,
                'summary of true class:',
                ydesc,
                ]
    elif isinstance(data, dataset.dstv):
        tdesc = summary(data.train)
        vdesc = summary(data.valid)
        desc = ['summary of train set:',
                tdesc,
                'summary of validation set:',
                vdesc,
                ]
    elif isinstance(data, dataset.desc):
        desc = '  ' + str(data)
    elif _is_small_data(data):
        desc = '  value: ' + str(data)
    else:
        desc = _summary(data)
    return desc

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
        if d:
            s = CateDesc(d)
        else:
            s = None
    elif isinstance(data, pd.Series):
        d = desc_cate_series(data)
        if d is not None:
            name = data.name or 'data'
            s = pd.DataFrame(dict(name=d)).transpose()
        else:
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
        ret = None
    return ret

def _summary(data):
    if data is None:
        return '  data is NONE'

    ret = ['size:', str(data.shape)]

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray):
        if len(data.shape) <= 2:
            df = pd.DataFrame(data)
        else:
            # TODO too large, skip it as workaround
            return ret
            #raise ValueError('data with more than 2 dimension is not supported yet')
            #df = pd.DataFrame(data.reshape(data.shape[0], -1))
            df = pd.DataFrame(data.flatten())
    else:
        raise ValueError('unknown data type: %s' % type(data))


    dh = df.head(5)
    dn = df.describe().transpose()
    dc = desc_cate(df)
    ret.extend(['head n:', str(dh), \
            'take as numeric type:', str(dn)])
    ret.append('take as category type:')
    if dc:
        ret.append(str(dc))
    else:
        ret.append('  seems not to be categorical')
    return ret

def _is_small_data(data):
    ret = isinstance(data, (int, float, basestring)) \
            or (isinstance(data, (list, tuple, dict, set)) and len(data) < 10)
    return ret

class CateDesc(object):
    def __init__(self, d):
        self._data = d

    def __repr__(self):
        return __str__

    def __str__(self):
        #print self._data
        str_list = [ '......' if n == '...' else str(pd.DataFrame({n:s}).transpose()) \
                for n, s in self._data ]
        return '\n'.join(str_list)

def main():
    return

if __name__ == '__main__':
    main()
