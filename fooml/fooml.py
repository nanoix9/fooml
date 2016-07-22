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


class FooML(object):

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

    def use_data(self, data, **kwds):
        name = data
        ds = dataset.load_data(data, **kwds)
        self.add_data(ds, name=name)

    def load_csv(self, name, path, **opt):
        ds = self._get_data_from_cache(name)
        if ds is None:
            self._report('cache missed for data "{}", load original data'.format(name))
            ds = dataset.load_csv(path, **opt)
            self._set_data_to_cache(name, ds)
            self._report('data "%s" is cached' % name)
        else:
            self._report('load data "{}" from cache'.format(name))
        self.add_data(ds, name=name)

    def load_image_grouped(self, name, path=None, train_path=None, test_path=None, **opt):
        ds = self._get_data_from_cache(name)

        if ds is None:
            self._report('cache missed for data "{}", load original data'.format(name))
            if path is not None:
                train_path = os.path.join(path, 'train')
                test_path = os.path.join(path, 'test')
            #ds = dataset.load_image(image_path, target_path, sample_id, target)
            ds_train = dataset.load_image_grouped(train_path, **opt)
            if test_path is not None:
                ds_test = dataset.load_image_flat(test_path, **opt)
                ds = (ds_train, ds_test)
            else:
                ds_test = None
                ds = ds_train
            self._set_data_to_cache(name, ds)
            self._report('data "%s" is cached' % name)
        else:
            self._report('load data "{}" from cache'.format(name))

        self.add_data(ds, name=name)

    def add_data(self, data, test=None, name='data'):
        if isinstance(data, (str, unicode)):
            name = data
            ds = dataset.load_data(data)
        else:
            ds = data

        if name in self._ds_train:
            self._report('Warning: Dataset with name "%s" already exists. Will be replaced' % name)

        if isinstance(ds, tuple):
            self._ds_train[name] = ds[0]
            self._ds_test[name] = ds[1]
        else:
            self._ds_train[name] = ds
        #print self._ds_train

    def load_data_now(self):
        for func in self._data_load_routine:
            func()

    def get_train_data(self, name):
        return self._ds_train[name]

    def enable_data_cache(self, cache_dir=None):
        self._data_cache = cache.DataCache(self._name, cache_dir)
        self._use_data_cache = True
        self._report('data cache enabled: %s' % self._data_cache._get_path(''))

    def _get_data_from_cache(self, name):
        if self._use_data_cache:
            self._report('load from cache: "{}"'.format(name))
            return self._data_cache.get(name)
        return None

    def _set_data_to_cache(self, name, data):
        if self._use_data_cache:
            return self._data_cache.set(name, data)

    def get_comp(self, name):
        return self._graph.get_comp(name)

    def add_comp(self, name, acomp, inp, out):
        self._graph.add_comp(name, acomp, inp, out)
        return self

    def _add_new_comp(self, name, acomp, inp, out, package=None, args=[], opt={}, comp_opt={}):
        if isinstance(acomp, basestring):
            #package, acomp_name = acomp.split(':')
            c = factory.create_comp(acomp, package=package, args=args, opt=opt, comp_opt=comp_opt)
        elif isinstance(acomp, comp.Comp):
            c = acomp
        else:
            c = factory.obj2comp(acomp, comp_opt)
        self.add_comp(name, c, inp, out)
        return self

    def add_split(self, name, input, output, args=[], opt={}, partition=None, part_key=None):
        comp_opt = {}
        if partition is not None:
            real_inp = slist.to_list(input)
            real_inp.append(partition)
            if part_key is not None:
                comp_opt['part_key'] = part_key
            spl = factory.create_comp('partsplit', args=args, opt=opt, comp_opt=comp_opt)
        else:
            real_inp = input
            spl = factory.create_comp('split', args=args, opt=opt, comp_opt=comp_opt)
        self.add_comp(name, spl, real_inp, output)

    def add_trans(self, name, acomp, input, output, args=[], opt={}, comp_opt={}):
        self._add_new_comp(name, acomp, input, output, args=args, opt=opt, comp_opt={})
        return self

    def add_feat_trans(self, name, obj, input, output, args=[], opt={}, comp_opt={}):
        if isinstance(obj, comp.Comp):
            self.add_comp(name, obj, input, output)
        elif hasattr(obj, '__call__'):
            self.add_comp(name, misc.FeatTransComp((obj, args, opt), **comp_opt), input, output)
        else:
            raise TypeError()

    def add_inv_trans(self, name, another, input, output):
        acomp = self.get_comp(another)
        inv_comp = factory.create_inv_trans(acomp)
        self.add_comp(name, inv_comp, input, output)
        return self

    def add_classifier(self, name, acomp, input, output=__NULL, proba=None):
        self._add_new_comp(name, acomp, input, output, package='sklearn', comp_opt=dict(proba=proba))
        return self

    def add_nn(self, name, nn, input, output=__NULL, train_opt={}):
        ''' Add nerual networks '''

        acomp = factory.obj2comp(nn, comp_opt=dict(train_opt=train_opt))
        self.add_comp(name, acomp, input, output)

    def evaluate(self, indic, input, acomp=None):
        if acomp is not None:
            self.add_comp(indic, acomp, input, FooML.__NULL)
        else:
            for i in slist.iter_multi(indic):
                #eva = factory.create_evaluator(i)
                #self.add_comp(i, eva, input, FooML.__NULL)
                self._add_new_comp(i, i, input, FooML.__NULL)
        return self

    def save_output(self, outs, path=None, opt={}):
        self._outputs.extend(slist.iter(outs))
        for out in slist.iter(outs):
            self._output_opts[out] = (path, opt)
        return self

    def set_target(self, target):
        self._target = target

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


def __test1():
    foo = FooML('__test1')
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
    foo.evaluate('AUC', input='y.lr')

    #foo.add_trans('decide', 'decide', input='y.lr', output='y.lr.c')
    foo.evaluate('report', input='y.lr.c')
    #foo.save_output(['y.lr', 'y.lr.c'])
    #foo.save_output('y.lr')

    foo.compile()
    foo.run_train()

def main():
    __test1()
    return

if __name__ == '__main__':
    main()
