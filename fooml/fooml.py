#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
#import collections as c
import wrap
import dataset
import stats
import report


class FooML(object):

    def __init__(self):
        self._reporter = report.TxtReporter()
        self._err = sys.stderr
        self._ds = {}
        self._comp = []
        self._target = None

    def add_data(self, data, test=None, name='data'):
        if isinstance(data, (str, unicode)):
            name = data
            ds = dataset.load_data(data)

        if name in self._ds:
            self._report('Warning: Dataset with name "%s" already exists. Will be replaced' % name)
        self._ds[name] = ds

    def add_component(self, obj, name=None, func=None):
        self._comp.append((name, obj))

    def add_classifier(self, clf, name='clf'):
        if isinstance(clf, (str, unicode)):
            name = clf
            clf = wrap.create_classifier(clf)
        self.add_component(clf, name)

    def set_target(self, target):
        self._target = target

    def show(self):
        self._report('Fooml description:')

    def run(self):
        self.show()
        self.desc_data()
        self.run_train()
        self.run_test()

    def desc_data(self):
        self._report('Quick Summary of Original Data')
        self._report_leveldown()
        for name, ds in self._ds.iteritems():
            self._report('Summary of data set %s:' % name)
            self._report_leveldown()
            #self._report('train set of %s:' % name)
            self._desc(ds)
            #self._report('test set of %s:' % name)
            #self._desc(test)
            self._report_levelup()
        self._report_levelup()

    def run_train(self):
        self._report('training ...')
        self._report_leveldown()
        for name, obj in self._comp:
            self._report('training %s ...' % name)
            self._train_one(obj)
        self._report_levelup()

    def run_test(self):
        self._report('run testing ...')

    def _report_levelup(self):
        self._reporter.levelup()

    def _report_leveldown(self):
        self._reporter.leveldown()

    def _report(self, msg):
        self._reporter.report(msg)

    def _desc(self, data):
        if data is None:
            self._report('it\'s NULL')
            return
        desc = stats.summary(data)
        self._report(desc)

    def _train_one(self, obj):
        pass


def __test1():
    foo = FooML()
    foo.add_data('iris')
    foo.add_classifier('LR')
    foo.run()

def main():
    __test1()
    return

if __name__ == '__main__':
    main()
