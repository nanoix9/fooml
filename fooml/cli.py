#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("square", type=int,
    #                    help="display a square of a given number")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="enter debug mode")
    parser.add_argument("-D", "--desc-data", action="store_true",
                        help="describe data set for each input/output")
    return parser

def parse_args(parser=None, argv=None):
    #print sys.argv
    if parser is None:
        parser = get_parser()
    if argv is None:
        argv = sys.argv[1:]
    #print argv
    #sys.exit()
    return parser.parse_args(argv)

def test1():
    argv = '-v -d'.split()
    print argv, parse_args(argv=argv)
    argv = '-vvv -d'.split()
    print argv, parse_args(argv=argv)

def main():
    test1()
    return

if __name__ == '__main__':
    main()
