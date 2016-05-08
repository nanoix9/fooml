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
from dt import slist


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
        self._outputs = []

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

    def add_comp(self, name, acomp, inp, out):
        self._comp.add_comp(name, acomp, inp, out)
        return self

    def add_comp_with_creator(self, name, acomp, inp, out, creator=None):
        if isinstance(acomp, basestring):
            clf = creator(acomp)
        else:
            clf = acomp
        self.add_comp(name, clf, inp, out)
        return self

    def add_ds_trans(self, name, acomp, input, output):
        self.add_comp_with_creator(name, acomp, input, output, factory.create_trans)
        return self

    def add_classifier(self, name, acomp, input, output=__NULL):
        self.add_comp_with_creator(name, acomp, input, output, factory.create_classifier)
        return self

    def evaluate(self, indic, pred, acomp=None):
        if acomp is not None:
            self.add_comp(indic, acomp, pred, FooML.__NULL)
        else:
            for i in slist.iter_multi(indic):
                eva = factory.create_evaluator(i)
                self.add_comp(i, eva, pred, FooML.__NULL)
        return self

    def save_output(self, outs):
        self._outputs.extend(slist.iter_multi(outs))
        return self

    def set_target(self, target):
        self._target = target

    def show(self):
        self._report('Fooml description:')
        self._report('Graph of computing components: %s' % self._comp)

    def run(self):
        self._comp.set_input(util.key_or_keys(self._ds))
        #self._comp.set_output(self._outputs + [FooML.__NULL])
        if self._outputs:
            self._comp.set_output(self._outputs)
        else:
            self._comp.set_output(FooML.__NULL)

        self.show()
        self.desc_data()

        self._report('Compiling graph ...')
        self._exec.compile_graph(self._comp)

        self._report('Training ...')
        out = self._exec.run_train(self._ds, data_keyed=True)

        self._report('Run Testing ...')
        ds = { k: v.X for k, v in self._ds.iteritems() }
        out = self._exec.run_test(ds, data_keyed=True)
        print 'final output:\n', out

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
    foo.add_ds_trans('binclass', 'binclass', input='iris', output='iris.2')
    foo.add_classifier('lr', 'LR', input='iris.2', output='y.lr')
    #foo.add_classifier('RandomForest', input='x')
    #foo.cross_validate('K', k=4)
    foo.evaluate('AUC', pred=['y.lr'])
    foo.save_output('y.lr')
    foo.run()

def main():
    __test1()
    return

if __name__ == '__main__':
    main()
