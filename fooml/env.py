#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import dataset
from log import logger

__env = {}

_curr_self = None
_curr_input = None
_curr_output = None

def self(action=lambda x: x):
    return lambda: action(_curr_self)

feat_dim = lambda: dataset.get_train(_curr_input)[0].shape[1]
out_dim = lambda: dataset.get_train(_curr_output)[0].shape[1]

def set_env(key, val, data=None, out=None):
    if hasattr(val, '__call__'):
        val = val()
    logger.info('set global environment %s=%s' % (str(key), str(val)))
    __env[key] = val

def get_env(key):
    return __env[key]

def get_all():
    return __env

def main():
    return

if __name__ == '__main__':
    main()
