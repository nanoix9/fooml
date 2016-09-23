#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import fooml
from fooml import env
from fooml.log import logger


def main():
    foo = fooml.FooML('bosch')
    foo.set_data_home('/vola1/scndof/data/bosch')
    #foo.enable_data_cache()

    foo.load_csv('ds_num', train_path='train_numeric.csv', target='Response', opt=dict(index_col='Id', nrows=100))

    foo.compile()
    foo.run_train()
    return

if __name__ == '__main__':
    main()
