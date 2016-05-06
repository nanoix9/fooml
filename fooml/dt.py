#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

class slist(object):

    def __init__(self, obj=None):
        self.__obj = obj

    def __getitem__(self, i):
        return slist.get(self.__obj, i)

    def __iter__(self):
        return slist.iter_multi(self.__obj)

    @staticmethod
    def to_list(obj, copy=False):
        if slist._is_coll(obj):
            if copy:
                return list(obj)
            else:
                return obj
        else:
            return [obj]

    @staticmethod
    def call_obj(obj, func, *args):
        if slist._is_coll(obj):
            return func(obj, *args)
        else:
            return None

    @staticmethod
    def index(obj, elem):
        return slist.call_obj(obj, list.index, elem)

    @staticmethod
    def str_index(idx):
        if idx is None:
            return ''
        else:
            return '[%s]' % idx

    @staticmethod
    def nones_like(obj):
        return slist.rep_like(None, obj)

    @staticmethod
    def ones_like(obj):
        return slist.rep_like(1, obj)

    @staticmethod
    def rep_like(v, obj):
        return slist.map(lambda x: v, obj)

    @staticmethod
    def get(obj, idx):
        if idx is None:
            return obj
        else:
            if not slist._is_coll(obj):
                raise ValueError('index is not None but object is not a list or tuple')
            return obj[idx]

    #def enumerate_maybe_list(obj, *args):
    @staticmethod
    def enumerate_multi(obj, *args, **kwds):
        #print 'iter_multi args:', obj, args
        strict = kwds.get('strict', False)
        if slist._is_coll(obj, strict):
            if any([len(a) != len(obj) for a in args]):
                raise ValueError('length of lists are not identical')
            for i, o in enumerate(obj):
                yield tuple([i, o] + [a[i] for a in args])
        else:
            yield tuple((None, obj) + args)

    @staticmethod
    def iter_multi(obj, *args, **kwds):
        #print 'iter_multi args:', obj, args
        strict = kwds.get('strict', False)
        if slist._is_coll(obj, strict):
            if any([len(a) != len(obj) for a in args]):
                raise ValueError('length of lists are not identical')
            for i, o in enumerate(obj):
                yield stuple.norm([o] + [a[i] for a in args])
        else:
            yield stuple.norm((obj,) + args)

    @staticmethod
    def map(func, obj):
        if slist._is_coll(obj):
            #print func, obj
            return map(func, obj)
        else:
            return func(obj)

    @staticmethod
    def _is_coll(obj, strict=False):
        if strict:
            return isinstance(obj, list)
        else:
            return isinstance(obj, (tuple, list))

class stuple(object):

    @staticmethod
    def norm(obj):
        if slist._is_coll(obj):
            if len(obj) > 1:
                return tuple(obj)
            else:
                return obj[0]
        else:
            return obj


######## tests

def test_slist():
    print [ a for a in slist.iter_multi('a')]
    print [ a for a in slist.iter_multi(['a', 1, 2])]
    print [ a for a in slist.iter_multi(['a', 1, 2], [10, 20, 30])]
    print [ a for a in slist.iter_multi(['a', 1], [10, 20], [100, 110])]
    #print [ a for a in slist.iter_multi(['a', 1, 2], [10, 20])]

def test_slist2():
    print slist.ones_like('ab')
    print slist.ones_like(list('ab'))
    print slist.index(['a', 0, 1], 1)

def main():
    test_slist()
    test_slist2()
    return

if __name__ == '__main__':
    main()
