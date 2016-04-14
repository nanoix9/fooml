#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp


class Executor(object):

    def __init__(self, reporter):
        self._reporter = reporter
        #self._comp = comp.Serial()

    def add_component(self, obj, name=None, func=None):
        if not isinstance(obj, comp.Component):
            c = comp.Component(name, obj)
        else:
            c = obj
        self._comp.add_component(c)

    def run_train(self, acomp, start_data):
        data = start_data
        self._report_leveldown()
        self._train_component(acomp, data)
        self._report_levelup()

    def _train_component(self, acomp, data):
        if isinstance(acomp, comp.Parallel):
            self._report('training parallel "%s" ...' % acomp.name)
            self._report_leveldown()
            for c in acomp:
                d = self._train_component(c, data)
            self._report_levelup()
        elif isinstance(acomp, comp.Serial):
            self._report('training serial "%s" ...' % acomp.name)
            self._report_leveldown()
            d = data
            for c in acomp:
                d = self._train_component(c, d)
            self._report_levelup()
        else:
            #print acomp
            self._report('training basic "%s" ...' % acomp.name)
            d = self._train_one(acomp, data)

    def _train_one(self, obj, data):
        pass

    def _report_levelup(self):
        self._reporter.levelup()

    def _report_leveldown(self):
        self._reporter.leveldown()

    def _report(self, msg):
        self._reporter.report(msg)


def main():
    return

if __name__ == '__main__':
    main()
