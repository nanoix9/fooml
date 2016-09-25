#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import inspect
import pandas as pd

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

def joins(str_list, sep='\n', ind=2):
    if isinstance(str_list, basestring):
        return str_list

    limit = 10000
    ll = []
    for s in str_list:
        #print s
        if isinstance(s, basestring):
            ll.append(s)
        elif isinstance(s, (list, tuple)):
            ll.append(indent(joins(s, sep, ind), ind))
        else:
            txt = str(s)
            if (len(txt) > limit):
                txt = txt[:limit] + '\n' + '......(string too long)'
            ll.append(txt)
            #raise TypeError('element must be basestring, list or tuple')
    #print ll
    #sys.exit()
    return sep.join(ll)

def limit_lines(text, limit=50):
    arr = text.strip().split('\n')
    if len(arr) > limit:
        lines = list(arr[0:limit/2])
        lines.append('...')
        lines.extend(arr[-(limit/2+1):])
        text = '\n'.join(lines)
    return text

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

############ Keras #############
def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


######### pandas #######
def get_index_or_col(df, name):
    #print df, name
    #print df.index.names
    if isinstance(name, basestring) and name in df.index.names:
        return df.index.get_level_values(name)
    else:
        return df[name]

def get_index_uniq_values(df, name):
    if name is None:
        return df.index
    elif isinstance(df.index, pd.MultiIndex):
        return pd.Index(df.index.levels[df.index.names.index(name)], name=name)
    elif isinstance(df.index, pd.Index):
        if df.index.name == name:
            return df.index
        else:
            raise RuntimeError('index name does not match')
    else:
        raise TypeError()

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
