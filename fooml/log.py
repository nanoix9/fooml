#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging


logger = logging.getLogger("FooML")
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
#fh = logging.FileHandler("spam.log")
#fh.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(name)-12s %(asctime)s %(levelname)-8s %(message)s', '%a, %d %b %Y %H:%M:%S',)
#fh.setFormatter(formatter)
#logger.addHandler(fh)

# create console handler
__ch = logging.StreamHandler()
__ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
__formatter = logging.Formatter("%(asctime)s [%(filename)s:%(funcName)s] %(levelname)s - %(message)s",
        '%Y-%m-%d %H:%M:%S')
__ch.setFormatter(__formatter)
logger.addHandler(__ch)

# add the handlers to logger

def main():
    logger.debug('debug')
    logger.error('error')
    return

if __name__ == '__main__':
    main()
