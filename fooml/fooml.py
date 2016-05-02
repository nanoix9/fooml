#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
#import collections as c
import wrap
import dataset
import stats
import report
import executor
#import comp
import graph
import factory
import util


class FooML(object):

    __NULL = '_'

    def __init__(self):
        self._reporter = report.TxtReporter()
        self._err = sys.stderr
        self._ds = {}
        #self._comp = comp.Serial()
        self._comp = graph.CompGraph('main')
        self._exec = executor.Executor(self._reporter)
        self._target = None

    def use_data(self, data):
        name = data
        ds = dataset.load_data(data)
        self.add_data(ds, name=name)

    def add_data(self, data, test=None, name='data'):
        if isinstance(data, (str, unicode)):
            name = data
            ds = dataset.load_data(data)
        else:
            ds = data

        if name in self._ds:
            self._report('Warning: Dataset with name "%s" already exists. Will be replaced' % name)
        self._ds[name] = ds
        #print self._ds

    def add_comp(self, acomp, name, inp, out):
        return self._comp.add_comp(name, acomp, inp, out)

    def add_classifier(self, name, input, output=__NULL, comp=None):
        if comp is None:
            clf = factory.create_classifier(name)
        else:
            clf = comp
        self.add_comp(clf, name, input, output)

    def set_target(self, target):
        self._target = target

    def show(self):
        self._report('Fooml description:')
        self._report('Graph of computing components: %s' % self._comp)

    def run(self):
        self._comp.set_input(util.key_or_keys(self._ds))
        self._comp.set_output(FooML.__NULL)

        self.show()
        self.desc_data()

        self._report('Compiling graph ...')
        self._exec.compile_graph(self._comp)

        self._report('Training ...')
        self._exec.run_train(self._ds, data_keyed=True)

        self._report('Run Testing ...')
        ds = { k: v.X for k, v in self._ds.iteritems() }
        self._exec.run_test(ds, data_keyed=True)

    def desc_data(self):
        self._report('Quick Summary of Original Data')
        self._report_leveldown()
        for name, ds in self._ds.iteritems():
            self._report('Summary of data set "%s":' % name)
            self._report_leveldown()
            #self._report('train set of %s:' % name)
            self._desc(ds)
            #self._report('test set of %s:' % name)
            #self._desc(test)
            self._report_levelup()
        self._report_levelup()


    def run_test(self):
        pass

    def _desc(self, data):
        if data is None:
            self._report('it\'s NULL')
            return
        desc = stats.summary(data)
        self._report(desc)

    def _report_levelup(self):
        self._reporter.levelup()

    def _report_leveldown(self):
        self._reporter.leveldown()

    def _report(self, msg):
        self._reporter.report(msg)


def __test1():
    foo = FooML()
    foo.use_data('iris')
    #foo.add_cutter('adapt', input='iris', output='cutted')
    #foo.add_fsel('Kbest', input='cutted', output='x')
    foo.add_classifier('LR', input='iris')
    #foo.add_classifier('RandomForest', input='x')
    #foo.cross_validate('K', k=4)
    #foo.evaluate('AUC')
    foo.run()

def main():
    __test1()
    return

if __name__ == '__main__':
    main()
