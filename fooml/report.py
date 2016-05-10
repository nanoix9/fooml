#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
from log import logger
import logging
import util

class Reporter(object):

    def __init__(self, prefix='  '):
        self.level = 1
        #self.__init_indent = ' ' * 8
        self.__init_indent = ''
        self.prefix = prefix

    def levelup(self):
        self.level -= 1
        self._check_level(self.level)
        self._block_end()

    def leveldown(self):
        self.level += 1
        self._block_begin()

    def setlevel(self, level):
        self._check_level(level)
        self._block_end()
        self.level = level
        self._block_begin()

    def report(self, msgs):
        if hasattr(msgs, '__iter__'):
            for m in msgs:
                self._report_recur(m)
        else:
            self._report_recur(msgs)

    def _report_recur(self, msgs):
        if hasattr(msgs, '__iter__'):
            self.leveldown()
            for m in msgs:
                self._report_recur(m)
            self.levelup()
        else:
            self.report_str(str(msgs))

    def _check_level(self, level):
        if level <= 0:
            raise ValueError('level should be greater than or equal 1')

    def _get_prefix(self):
        return self.prefix * (self.level - 1)

    def _block_begin(self):
        pass

    def _block_end(self):
        pass


class LogReporter(Reporter):

    def __init__(self, out=sys.stdout, prefix='  '):
        super(LogReporter, self).__init__(prefix)
        self._out = out

    def report_str(self, msg):
        __cf = logging.currentframe
        def __new_cf():
            f = __cf()
            name = f.f_code.co_name
            #print(name)
            while name != 'report_str':
                f = f.f_back
                name = f.f_code.co_name
            while name.startswith('report') or name.startswith('_report') \
                    or name.startswith('__report'):
                #print(name)
                f_prev = f
                f = f.f_back
                name = f.f_code.co_name
            return f_prev
        logging.currentframe = __new_cf

        #indent = self.__init_indent + prefix
        prefix = self._get_prefix()
        for line in msg.split('\n'):
            s = prefix + line
            logger.info('%s' % s)
        #print(s, file=self._out)

        logging.currentframe = __cf


class MdReporter(Reporter):

    __block_marks = {
            1: ('', '\n = = = \n'),
            2: ('', '\n - - - \n'),
            #1: ('======', ''),
            #2: ('------', ''),
            #3: ('###', ),
            #4: ('####', ),
        }

    def __init__(self, rep_file):
        super(MdReporter, self).__init__()
        if isinstance(rep_file, basestring):
            self._file = open(rep_file, 'w')
        self.level -= 1
        self.leveldown()  # start from level 1

    def _block_begin(self):
        if self.level in MdReporter.__block_marks:
            self._file.write(MdReporter.__block_marks[self.level][0] + '\n')

    def _block_end(self):
        if self.level in MdReporter.__block_marks:
            marks = MdReporter.__block_marks[self.level]
            if len(marks) > 1:
                self._file.write(marks[1] + '\n')

    def report_str(self, msg):
        self._file.writelines(util.indent(msg, prefix=self._get_prefix()) + '\n')

class SeqReporter(Reporter):

    def __init__(self):
        self._reporters = []

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def levelup(self):
        for r in self._reporters:
            r.levelup()

    def leveldown(self):
        for r in self._reporters:
            r.leveldown()

    def setlevel(self, level):
        for r in self._reporters:
            r.setlevel(level)

    def report(self, msgs):
        for r in self._reporters:
            r.report(msgs)


def main():
    return

if __name__ == '__main__':
    main()
