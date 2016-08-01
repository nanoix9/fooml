#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: add support for compiling & executing nested graph comp

#from __future__ import print_function

import sys
import comp
import graph
import collections
import util
from dt import slist
import report
import stats
from log import logger


class Executor(object):

    __INPUT__ = '__INPUT__'
    __OUTPUT__ = '__OUTPUT__'
    __NULL__ = '_'

    def __init__(self, reporter=None):
        self._reporter = reporter
        self._graph = None
        self._cgraph = None

    def set_reporter(self, reporter):
        self._reporter = reporter

    def set_graph(self, g):
        self._graph = g
        return self

    def compile(self):
        logger.info('Compiling graph "%s" ...' % self._graph.name)
        self._cgraph = graph._CompiledGraph(self._graph)
        for name, acomp in self._graph.iter_comps():
            if hasattr(acomp, '_exec') and isinstance(acomp._exec, Executor):
                acomp._exec.set_reporter(self._reporter)
                acomp._exec.compile()
        return self

    def compile_graph(self, g):
        return self.set_graph(g).compile()

    def show(self):
        if self._cgraph is None:
            logger.warning('no graph is compiled yet')
            return

        self._report('Executor for tasks:')
        #print str(self._cgraph)
        self._report(str(self._cgraph))

    def __str__(self):
        return '%s(\n%s\n)' % (util.get_type_fullname(self), \
                util.indent(str(self._cgraph), 2))

    def _desc_data(self, data, names):
        if isinstance(names, (list, tuple)):
            #self._report('summary of data "%s":' % str(names))
            for i, d in enumerate(data):
                self._desc_data(d, names[i])
        else:
            s = stats.summary(data)
            self._report('summary of data "%s":' % names)
            # self._report_leveldown()
            self._report(s)
            # self._report_levelup()

    def __get_data_list(self, data, keyed):
        if keyed:
            data_list = self.__data_dict_to_list(data, self._graph._inp)
        else:
            data_list = data
        return data_list

    def __data_dict_to_list(self, data_dict, input_names):
        #print data_dict
        return slist.map(lambda n: data_dict[n], input_names)

    def validate(self):
        if self._cgraph is None or self._graph is None:
            raise RuntimeError('no graph yet')

    def run_train(self, start_data, data_keyed=False):
        self.validate()
        data = self.__get_data_list(start_data, data_keyed)
        return self._train_comp(self._cgraph, data)

    def _train_comp(self, acomp, data):
        out = self._run_iter(acomp, data, self._train_one)
        return out

    def run_test(self, start_data, data_keyed=False):
        self.validate()
        data = self.__get_data_list(start_data, data_keyed)
        return self._test_comp(self._cgraph, data)

    def _test_comp(self, acomp, data):
        out = self._run_iter(acomp, data, self._test_one)
        return out

    def _train_one(self, basic_comp, data):
        #self._desc_data(data, self._graph.)
        out = basic_comp.fit_trans(data)
        #self._desc_data(out)
        return out

    def _test_one(self, basic_comp, data):
        #self._desc_data(data)
        out = basic_comp.trans(data)
        #self._desc_data(out)
        return out

    def _run_iter(self, acomp, data, func):
        fname = func.__name__
        if isinstance(acomp, graph._CompiledGraph):
            self._report('run "%s" across graph compiled "%s"' \
                % (fname, acomp._graph.name))
            out = self.run_compiled(data, func)
        elif isinstance(acomp, graph.CompGraph):
            #self._report('run "%s" across graph "%s"' \
            #    % (fname, acomp.name))
            ## TODO: refact this
            #out = self._train_comp(acomp, data)
            raise NotImplementedError()
        elif isinstance(acomp, comp.Comp):
            #print acomp
            self._report('run "%s" on component:\n%s' \
                    % (fname, util.indent(repr(acomp))))
            out = func(acomp, data)
        else:
            raise TypeError('unknown component type: "%s"' % acomp.__class__)
        return out

    def run_compiled(self, data, func):
        self._report('Run Compiled Graph "%s" ...' % self._cgraph._graph.name)
        task_seq = self._cgraph._task_seq

        if task_seq is None:
            raise ValueError('No compiled graph found')

        # create buffers for storing real inputs of each task
        logger.debug('creating input buffer ...')
        input_buff = self._create_input_buff()
        #print 'input buff', input_buff

        pending = collections.deque(i for i, _ in enumerate(task_seq))

        # run input
        curr_task_no = pending.popleft()
        c_name, entry = task_seq[curr_task_no]
        if c_name != Executor.__INPUT__:
            raise ValueError('First task should be input!')
        self._report('Task %d Input: assign input data' % curr_task_no)
        self._report_leveldown()
        self._emit_data_by_index(data, curr_task_no, input_buff)
        self._report_levelup()

        while pending:
            curr_task_no = pending.popleft()
            c_name, c_entry = task_seq[curr_task_no]
            curr_input = input_buff[curr_task_no]
            if not self._is_input_ready(curr_input):
                raise ValueError('Task %s does not recieve all input data' % c_name)
            if c_name == Executor.__OUTPUT__:
                self._report('Task %d Ouput: assign output results' % curr_task_no)
                self._report_leveldown()
                self._desc_data(curr_input, self._graph._out)
                ret = curr_input
                self._report_levelup()
            else:
                c_obj, c_inp, c_out = c_entry
                self._report('Task %d: train component "%s", input=%s, output=%s' \
                    % (curr_task_no, c_name, c_inp, c_out))
                self._report_leveldown()
                self._report('Summary of input of "%s": %s' % (c_name, c_inp))
                self._desc_data(curr_input, c_inp)
                out = self._run_iter(c_obj, curr_input, func)
                self._report('Summary of output of "%s": %s' % (c_name, c_out))
                self._desc_data(out, c_out)
                self._emit_data_by_index(out, curr_task_no, input_buff)
                input_buff[curr_task_no] = None  # clean input data
                self._report_levelup()
        return ret

    def _create_input_buff(self):
        def _iter_task_inp():
            yield self._graph._inp
            for cname, (c_obj, c_inp, c_out) in self._cgraph._task_seq[1:-1]:
                yield c_inp
            yield self._graph._out
        buff = [ slist.nones_like(inp) for inp in _iter_task_inp() ]
        return buff

    def _is_input_ready(self, buff):
        #print '-----> _is_input_ready:', buff
        return all([ d is not None for d in slist.iter_multi(buff)])

    def _emit_data_by_index(self, data, task_no, input_buff):
        oimap = self._cgraph._oimap[task_no]
        for o, c, i in oimap:
            self._report('emit data "%s".%s -> "%s".%s' \
                    % (self._cgraph.format_comp(task_no), \
                       self._cgraph.format_output(task_no, o), \
                       self._cgraph.format_comp(c), \
                       self._cgraph.format_input(c, i)))
            di = slist.get(data, o)
            if i is None:
                input_buff[c]  = di
            else:
                input_buff[c][i] = di

    def _report_levelup(self):
        self._reporter.levelup()

    def _report_leveldown(self):
        self._reporter.leveldown()

    def _report(self, msg):
        self._reporter.report(msg)

def test_exec():
    gcomp = graph.CompGraph('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', comp.PassComp(), 'x', 'u')
    gcomp.add_comp('c2', comp.ConstComp(1), ['input', 'u'], 'z')
    #gsub1 = graph.CompGraph('subgraph1', inp='s1', out='y1')
    #gsub1.add_comp('c31', comp.PassComp(), 's1', 'y1')
    #gcomp.add_comp('g3', gsub1, 'z', 'y')
    gcomp.add_comp('g3', comp.PassComp(), 'z', 'y')
    print gcomp

    exe = Executor(report.LogReporter())
    data = [(11,22), {'a':100, 10:200}]
    #exe.run_train(data, gcomp)
    exe.compile_graph(gcomp)
    exe.show()
    exe.run_train(data)

def test_nest():
    import comp.group as group
    gcomp = graph.CompGraph('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', comp.PassComp(), 'x', 'u')
    gcomp.add_comp('c2', comp.ConstComp(1), ['input', 'u'], 'z')

    gsub1 = graph.CompGraph('subgraph1', inp='s1', out='y1')
    gsub1.add_comp('c31', comp.PassComp(), 's1', 'y1')
    esub = Executor(report.LogReporter())
    esub.set_graph(gsub1)
    print gsub1

    gcomp.add_comp('c3', group.Seq(esub), 'z', 'y')
    print gcomp

    exe = Executor(report.LogReporter())
    exe.compile_graph(gcomp)

    data = [(11,22), {'a':100, 10:200}]
    #exe.run_train(data, gcomp)
    exe.show()
    exe.run_train(data)


def main():
    #test_exec()
    test_nest()
    return

if __name__ == '__main__':
    main()
