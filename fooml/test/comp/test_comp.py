#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from fooml import comp
from fooml import env
from fooml.executor import Executor
from fooml.dataset import dsxy

def test_env():
    exe = Executor()

    c = comp.ConstComp(10)
    c.set_env('pi', 3.14)
    c.set_env('e', 2.72)
    c.set_env('const', env.self(lambda self:self._obj))

    c2 = comp.ConstComp(15)
    c2.set_env('const', env.self(lambda self:self._obj))
    c2.set_env('nb_feat', env.feat_dim)

    data = 1
    env._curr_self = c
    env._curr_input = data
    exe._train_one(c, data)
    print env.get_all()

    data = dsxy(np.array([range(3), range(2,5)]), None)
    env._curr_self = c2
    env._curr_input = data
    exe._train_one(c2, data)

    print env.get_all()


def main():
    test_env()
    return

if __name__ == '__main__':
    main()
