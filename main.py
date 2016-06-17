#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import fooml


def test():
    fooml.main()

def testnn():
    from fooml.exam import nn
    nn.main()

def main():
    #foo = fooml.FooML()
    #foo.run()
    test()
    #testnn()
    return


if __name__ == '__main__':
    main()
