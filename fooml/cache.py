#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import os
import os.path
import settings
from log import logger


class DataCache(object):

    def __init__(self, name, cache_dir=None):
        if cache_dir is None:
            cache_dir = settings.CACHE_DIR
        self._name = name
        self._cache_dir = cache_dir

    def set(self, key, data):
        path = self._get_path(key)
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            logger.info('create new directory for cache "%s": "%s"' % (self._name, dirname))
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        #else:
        #    raise RuntimeError('directory "%s" doesnt exists' % dirname)

    def get(self, key):
        path = self._get_path(key)
        data = None
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        return data

    def _get_path(self, key):
        return os.path.join(self._cache_dir, self._name, key)


def main():
    return

if __name__ == '__main__':
    main()

