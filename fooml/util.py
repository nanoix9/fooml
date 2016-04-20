#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


############ for dict ############
def key_or_keys(d):
    k = d.keys()
    if len(k) == 0:
        return None
    elif len(k) == 1:
        return k.pop()
    else:
        return list(k)

def gets_from_dict(adict, keys):
    ''' return a list while `keys` is a list or tuple of keys;
    or a value if `keys` is a single key
    '''

    if isinstance(keys, (tuple, list)):
        ret = [ adict[k] for k in keys ]
        if isinstance(keys, tuple):
            return tuple(ret)
        else:
            return ret
    else:
        return adict[keys]


############## for list ############
def to_list(obj, copy=False):
    if isinstance(obj, (tuple, list)):
        if copy:
            return list(obj)
        else:
            return obj
    else:
        return [obj]

def call_maybe_list(obj, func, *args):
    if isinstance(obj, (tuple, list)):
        return func(obj, *args)
    else:
        return None

def str_index(idx):
    if idx is None:
        return ''
    else:
        return '[%s]' % idx

def nones_like(obj):
    return rep_like_maybe_list(None, obj)

def ones_like(obj):
    return rep_like_maybe_list(1, obj)

def rep_like_maybe_list(v, obj):
    return map_maybe_list(lambda x: v, obj)

def get_maybe_list(obj, idx):
    if idx is None:
        return obj
    else:
        if not isinstance(obj, (tuple, list)):
            raise ValueError('index is not None but object is not a list or tuple')
        return obj[idx]

def enumerate_maybe_list(obj, *args):
    #print 'iter_maybe_list args:', obj, args
    if isinstance(obj, (tuple, list)):
        if any([len(a) != len(obj) for a in args]):
            raise ValueError('length of lists are not identical')
        for i, o in enumerate(obj):
            yield maybe_tuple([i, o] + [a[i] for a in args])
    else:
        yield maybe_tuple((None, obj) + args)

def iter_maybe_list(obj, *args):
    #print 'iter_maybe_list args:', obj, args
    if isinstance(obj, (tuple, list)):
        if any([len(a) != len(obj) for a in args]):
            raise ValueError('length of lists are not identical')
        for i, o in enumerate(obj):
            yield maybe_tuple([o] + [a[i] for a in args])
    else:
        yield maybe_tuple((obj,) + args)

def map_maybe_list(func, obj):
    if isinstance(obj, (tuple, list)):
        #print func, obj
        return map(func, obj)
    else:
        return func(obj)

########### for tuple ###########
def maybe_tuple(obj):
    if isinstance(obj, (tuple, list)):
        if len(obj) > 1:
            return tuple(obj)
        else:
            return obj[0]
    else:
        return obj

######### misc ##########
def replace_struct(obj, replace):
    if isinstance(obj, dict):
        return {replace_struct(k, replace): replace_struct(v, replace) \
                for k, v in obj.iteritems()}
    elif isinstance(obj, list):
        return [replace_struct(i, replace) for i in obj]
    elif isinstance(obj, tuple):
        return tuple([replace_struct(i, replace) for i in obj])
    else:
        return replace.get(obj, obj)


######## tests

def test_maybe_list():
    print [ a for a in iter_maybe_list('a')]
    print [ a for a in iter_maybe_list(['a', 1, 2])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20, 30])]
    print [ a for a in iter_maybe_list(['a', 1], [10, 20], [100, 110])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20])]

def test_replace_struct():
    s = {'a':[('a', 1), ['ab', 'a'], 'a', 100], 2:'a'}
    rep = {'a': 'x'}
    r = replace_struct(s, rep)
    print s
    print r


def test_dict():
    print key_or_keys(dict())
    print key_or_keys(dict(a=9))
    print key_or_keys(dict(a=1, b=2))

def main():
    test_dict()
    return

if __name__ == '__main__':
    main()
