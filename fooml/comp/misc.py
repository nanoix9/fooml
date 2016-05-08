#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from comp import FunComp
from fooml import dataset

class DsTransComp(FunComp):

    def __init__(self, fun_with_arg, on=['X', 'y']):
        super(DsTransComp, self).__init__(fun_with_arg)
        self._on = on

    def trans(self, data):
        X_out = data
        if self._is_exec('X'):
            X_out = self._call_func(data)
        return X_out

    def fit_trans(self, data):
        X, y = data
        X_out, y_out = X, y
        if self._is_exec('X'):
            X_out = self._call_func(X)
        if self._is_exec('y'):
            y_out = self._call_func(y)
        return dataset.dsxy(X_out, y_out)

    def _is_exec(self, name):
        return (isinstance(self._on, (list, tuple)) and name in self._on) \
                or self._on == name

    def _call_func(self, data):
        return super(DsTransComp, self).trans(data)


class TargTransComp(DsTransComp):

    def __init__(self, fun_with_arg):
        super(TargTransComp, self).__init__(fun_with_arg, 'y')


def main():
    return

if __name__ == '__main__':
    main()
