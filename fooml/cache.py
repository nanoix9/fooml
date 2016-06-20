#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import os.path
import settings


class DataCache(object):

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = settings.CACHE_DIR
        self._cache_dir = cache_dir

    def set(self, name, data):
        path = self._get_path(name)
        dirname = os.path.dirname(path)
        if os.path.isdir(dirname):
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise RuntimeError('directory "%s" doesnt exists' % dirname)

    def get(self, name):
        path = self._get_path(name)
        data = None
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        return data

    def _get_path(self, name):
        return os.path.join(self._cache_dir, name)


def main():
    return

if __name__ == '__main__':
    main()
