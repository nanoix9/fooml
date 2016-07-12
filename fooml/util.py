#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import inspect

def get_type_fullname(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__

def getmembers(obj, pre_exclude=['__']):
    return [a for a in inspect.getmembers(obj, \
                lambda a: not inspect.isroutine(a)) \
            if all([not a[0].startswith(p) for p in pre_exclude])]

########### for string ##########
def indent(s, ind=2, prefix=' '):
    str_ind = prefix * ind
    #print s
    #print prefix + s.replace('\n', '\n' + prefix)
    return str_ind + s.replace('\n', '\n' + str_ind)

############ for dict ############
def merge_dict_or_none(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        if dictionary is not None:
            result.update(dictionary)
    return result

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

def test_getmem():
    class _A(object):
        s = 100
        def __init__(self):
            self.a = 'aa'
            self.b = 100

    print getmembers(_A())

def main():
    #test_dict()
    test_getmem()
    return

if __name__ == '__main__':
    main()
