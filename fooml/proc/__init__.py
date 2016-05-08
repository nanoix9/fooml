#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np


def binclass(y, pos=lambda x: x > 0):
    return np.array([1 if pos(x) else 0 for x in y])
    #return np.apply_along_axis(pos, 0, y)


def test_binclass():
    a = np.arange(5)
    print a
    print binclass(a)

def main():
    test_binclass()
    return

if __name__ == '__main__':
    main()
