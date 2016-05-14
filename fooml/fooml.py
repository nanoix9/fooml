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
        self._reporter = report.SeqReporter()
        self._err = sys.stderr
        self._ds = {}
        #self._comp = comp.Serial()
        self._comp = graph.CompGraph('main')
        self._exec = executor.Executor(self._reporter)
        self._target = None
        self._outputs = []

    def add_reporter(self, reporter):
        self._reporter.add_reporter(reporter)

    def use_data(self, data, **kwds):
        name = data
        ds = dataset.load_data(data, **kwds)
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

    def add_comp_with_creator(self, name, acomp, inp, out, creator=None, **opt):
        if isinstance(acomp, basestring):
            clf = creator(acomp, **opt)
        else:
            clf = acomp
        self.add_comp(name, clf, inp, out)
        return self

    def add_trans(self, name, acomp, input, output):
        self.add_comp_with_creator(name, acomp, input, output, factory.create_trans)
        return self

    def add_classifier(self, name, acomp, input, output=__NULL, proba=None):
        self.add_comp_with_creator(name, acomp, input, output, factory.create_classifier, proba=proba)
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

    def run(self, test=True):
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

        if test:
            self._report('Run Testing ...')
            ds = { k: dataset.dsxy(v.X, None) for k, v in self._ds.iteritems() }
            out = self._exec.run_test(ds, data_keyed=True)
        print 'final output:\n', out

    def run_train(self):
        return self.run(test=False)

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
    foo.add_reporter(report.LogReporter())
    foo.add_reporter(report.MdReporter('report.md'))
    data_name = 'digits'
    data_name = 'iris'
    foo.use_data(data_name, flatten=True)

    #foo.add_cutter('adapt', input='iris', output='cutted')
    #foo.add_fsel('Kbest', input='cutted', output='x')
    iris_2 = 'iris.2'
    foo.add_trans('binclass', 'binclass', input=data_name, output=iris_2)
    #iris_2 = data_name

    #foo.add_classifier('lr', 'LR', input=iris_2, output='y.lr.c')
    #foo.add_classifier('lr', 'LR', input=iris_2, output='y.lr', proba='only')
    foo.add_classifier('lr', 'LR', input=iris_2, output=['y.lr.c', 'y.lr'], proba='with')

    #foo.add_classifier('clf', 'DecisionTree', input=iris_2, output='y.lr.c')
    #foo.add_classifier('lr', 'LR', input='iris', output='y.lr')
    #foo.add_classifier('RandomForest', input='x')

    #foo.cross_validate('K', k=4)
    foo.evaluate('AUC', pred='y.lr')

    #foo.add_trans('decide', 'decide', input='y.lr', output='y.lr.c')
    foo.evaluate('report', pred='y.lr.c')
    #foo.save_output(['y.lr', 'y.lr.c'])
    #foo.save_output('y.lr')
    foo.run_train()

def main():
    __test1()
    return

if __name__ == '__main__':
    main()
