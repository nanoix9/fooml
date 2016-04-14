#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys

class TxtReporter(object):

    def __init__(self, out=sys.stdout, prefix='  '):
        self._out = out
        self.level = 1
        self.prefix = prefix

    def levelup(self):
        self.level -= 1
        self._check_level(self.level)

    def leveldown(self):
        self.level += 1

    def setlevel(self, level):
        self._check_level(level)
        self.level = level

    def report(self, msgs):
        prefix = self.prefix * self.level
        self._report_recur(msgs, prefix)

    def _report_recur(self, msgs, prefix):
        if hasattr(msgs, '__iter__'):
            for m in msgs:
                self._report_recur(m, self.prefix + prefix)
        else:
            self.report_str(str(msgs), prefix)

    def report_str(self, msg, prefix):
        print(prefix + msg.replace('\n', '\n' + prefix), file=self._out)

    def _check_level(self, level):
        if level <= 0:
            raise ValueError('level should be greater than or equal 1')

def main():
    return

if __name__ == '__main__':
    main()
