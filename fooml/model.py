#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os.path
#import collections as c
import settings
import dataset
import cache
import stats
import report
import executor
import comp
from comp import misc
import graph
import factory
import util
from dt import slist
from log import logger



class Model(object):

    __NULL = '_'

    def __init__(self, name='fool'):
        self._name = name
        self._reporter = report.SeqReporter()
        self._err = sys.stderr
        self._ds_train = {}
        self._ds_test = {}
        #self._graph = comp.Serial()
        self._graph = graph.CompGraph(name)
        self._exec = executor.Executor(self._reporter)
        self._target = None
        self._outputs = []
        self._output_opts = {}
        self._use_data_cache = False
        self._out_dir = settings.OUT_DIR
        self._data_load_routine = []

        self.add_reporter(report.LogReporter())

    def add_reporter(self, reporter):
        self._reporter.add_reporter(reporter)

    def report_to(self, md_path):
        if md_path.endswith('.md'):
            self.add_reporter(report.MdReporter(md_path))
        else:
            raise ValueError('only support Markdown reporter yet')

    def get_comp(self, name):
        return self._graph.get_comp(name)

    def add_comp(self, name, acomp, inp, out):
        self._graph.add_comp(name, acomp, inp, out)
        return self

    def show(self):
        self._report('Fooml description:')
        self._report('Graph of computing components: %s' % self._graph)

    def compile(self):
        self._graph.set_input(util.key_or_keys(self._ds_train))
        #self._graph.set_output(self._outputs + [FooML.__NULL])
        if self._outputs:
            self._graph.set_output(self._outputs)
        else:
            self._graph.set_output(FooML.__NULL)

        self._report('Compiling graph ...')
        self._exec.set_graph(self._graph)

    def run(self, test=True):
        self.show()
        self._exec.show()

        self.desc_data()

        self._report('Training ...')
        out = self._exec.run_train(self._ds_train, data_keyed=True)

        if test:
            self._report('Run Testing ...')
            ds = self._get_test_data()
            out = self._exec.run_test(ds, data_keyed=True)

        if len(self._outputs) > 0:
            self._save_result(out)

    def run_train(self):
        return self.run(test=False)

    def _save_result(self, out_data):
        for i, ds in slist.enumerate(out_data):
            #print i, ds
            ds_name = slist.get(self._outputs, i)
            path, opt = self._get_opt_for_save(ds_name, 'csv')
            self._report('saving data "{}" to "{}"'.format(ds_name, path))
            dataset.save_csv(ds, path, opt)
        #print 'final output:\n', out

    def _get_opt_for_save(self, name, type):
        path, opt = self._output_opts[name]
        if not path:
            path = os.path.join(self._out_dir, name) + '.' + type
        return path, opt

    def _get_test_data(self):
        ds = {}
        for k, v in self._ds_train.iteritems():
            if self._ds_test.get(k, None) is not None:
                ds[k] = self._ds_test[k]
            else:
                ds[k] = dataset.dsxy(v.X, None, v.index)
        return ds

    def desc_data(self):
        self._report('Quick Summary of Original Data')
        self._report_leveldown()
        for name, ds in self._ds_train.iteritems():
            self._report('Summary of data set "%s":' % name)
            self._report_leveldown()
            self._report('train set of %s:' % name)
            self._desc(ds)
            self._report('test set of %s:' % name)
            self._desc(self._ds_test.get(name, None))
            self._report_levelup()
        self._report_levelup()


    def run_test(self):
        pass

    def _desc(self, data):
        if data is None:
            self._report('dataset is NULL')
            return
        desc = stats.summary(data)
        self._report(desc)

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
