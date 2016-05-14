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

    def __init__(self, reporter):
        self._reporter = reporter
        #self._comp = comp.Serial()
        self._oimap = None
        self._task_seq = None
        self._graph = None

    def add_component(self, obj, name=None, func=None):
        if not isinstance(obj, comp.Component):
            c = comp.Component(name, obj)
        else:
            c = obj
        self._comp.add_component(c)

    def _run_iter(self, acomp, data, func):
        fname = func.__name__
        if acomp is None:
            self._report('run "%s" across graph compiled "%s"' \
                % (fname, self._graph.name))
            out = self.run_compiled(data, func)
        elif isinstance(acomp, graph.CompGraph):
            self._report('run "%s" across graph "%s"' \
                % (fname, acomp.name))
            # TODO: refact this
            out = self._train_comp(acomp, data)
            #self.run_train(acomp, data)
        else:
            #print acomp
            self._report('run "%s" on basic component:\n%s' \
                    % (fname, util.indent(repr(acomp))))
            out = func(acomp, data)
        return out

    def __get_data_list(self, data, keyed):
        if keyed:
            data_list = self.__data_dict_to_list(data, self._graph._inp)
        else:
            data_list = data
        return data_list

    def run_train(self, start_data, acomp=None, data_keyed=False):
        data = self.__get_data_list(start_data, data_keyed)
        return self._train_comp(acomp, data)

    def _train_comp(self, acomp, data):
        out = self._run_iter(acomp, data, self._train_one)
        return out

    def run_test(self, start_data, acomp=None, data_keyed=False):
        data = self.__get_data_list(start_data, data_keyed)
        return self._test_comp(acomp, data)

    def _test_comp(self, acomp, data):
        out = self._run_iter(acomp, data, self._test_one)
        return out

    def dfs(self, graph, func):
        logger.info('start Depth-First Searching of graph %s ...' % graph.name)

        logger.debug('build fake input data to mark node visiting')
        data = slist.ones_like(graph._inp)
        logger.debug('fake input data: %s' % str(data))

        logger.debug('build input buffer for each edge and the final output')
        buff = self._graph_comp_to_input(graph)
        #print buff
        out_buff = {o: None for o in slist.iter_multi(graph._out)}

        logger.debug('setup input data to initialize graph searching')
        self._emit_data(data, graph._inp, graph, buff, out_buff)
        stack = slist.to_list(graph._inp, copy=True)
        visited = set()
        while stack:
            curr_node = stack.pop()
            logger.debug('visits node "%s"' % (curr_node))
            if curr_node in visited and curr_node != Executor.__NULL__:
                raise ValueError('node "%s" has already been visited' % curr_node)
            visited.add(curr_node)
            for f, t, comp_name, acomp in graph._edges_with_attr(curr_node, attr=('name', 'comp')):
                logger.debug('+ checking edge: %s -(%s)-> %s' % (f, comp_name, t)) #, acomp)
                curr_input = buff[comp_name]
                entry = graph._comps[comp_name]
                if self.__is_inputs_ready(curr_input):
                    logger.debug('+ edge "%s" is ready for visiting' % (comp_name,))
                    func((comp_name, entry))
                    out = slist.ones_like(entry.out)  # fake output data
                    self.__clear_inputs(curr_input)
                    self._emit_data(out, entry.out, graph, buff, out_buff)
                    stack.extend(slist.to_list(entry.out))
                    logger.debug('+ current output buffer of graph: %s' % out_buff)
        if any(d is None for n, d in out_buff.iteritems()):
            raise ValueError('Nothing is connected to output(s): %s' \
                % filter(lambda n: out_buff[n] is None, out_buff.keys()))
        ret = util.gets_from_dict(out_buff, graph._out)
        logger.debug('graph final output: %s' % ret)
        return ret

    def _train_graph(self, graph, data):
        raise NotImplementedError('')
        buff = self._graph_comp_to_input(graph)
        out_buff = {o: None for o in slist.iter_multi(graph._out)}
        print buff
        self._emit_data(data, graph._inp, graph, buff, out_buff)
        stack = slist.to_list(graph._inp, copy=True)
        while stack:
            curr_node = stack.pop()
            self._report('Dataset "%s" in graph "%s" is ready' \
                    % (curr_node, graph.name))
            for f, t, comp_name, acomp in graph._edges_with_attr(curr_node, attr=('name', 'comp')):
                print f, t, comp_name, acomp
                curr_input = buff[comp_name]
                entry = graph._comps[comp_name]
                if self.__is_inputs_ready(curr_input):
                    self._report('training component "%s" in graph "%s" ...' \
                            % (comp_name, graph.name))
                    real_input_data = self.__data_dict_to_list(curr_input, entry.inp)
                    print '>>> train:', acomp, real_input_data
                    out = self._train_comp(acomp, real_input_data)
                    print '>>> train out:', out
                    self.__clear_inputs(curr_input)
                    #out_names = slist.to_list(entry.out)
                    self._emit_data(out, entry.out, graph, buff, out_buff)
                    stack.extend(slist.to_list(entry.out))
                    print '>>> out of graph:', out_buff
        if any(d is None for n, d in out_buff.iteritems()):
            raise ValueError('Output did not get an value: %s' \
                % filter(lambda n: out_buff[n] is None, out_buff.keys()))
        ret = util.gets_from_dict(out_buff, graph._out)
        print('final output: %s' % ret)
        return ret

    def __is_inputs_ready(self, buff):
        return all([d is not None for i, d in buff.iteritems()])

    def __data_dict_to_list(self, data_dict, input_names):
        #print data_dict
        return slist.map(lambda n: data_dict[n], input_names)

    def __clear_inputs(self, buff):
        for k in buff:
            buff[k] = None

    def _emit_data(self, data, data_names, graph, buff, out_buff):
        #logger.debug('emit data "%s": %s' % (data, data_names))
        if any(d is None for d in slist.iter_multi(data)):
            raise ValueError('real data is none for data with name "%s"' % data_names)
        data_dict = { n:d for n, d in slist.iter_multi(data_names, data) }
        #print data_dict
        #sys.exit()
        #logger.debug('buffer before emit: %s' % buff)
        for dname, d_obj in data_dict.iteritems():
            emitted = False
            if dname in out_buff:
                if dname != Executor.__NULL__ and out_buff[dname] is not None:
                    raise ValueError('output of "%s" already got a value' % dname)
                logger.debug('emit to output "%s": %s' % (dname, d_obj))
                out_buff[dname] = d_obj;
                emitted = True
            for f, t, comp_name in graph._edges_with_attr(nbunch=[dname]):
                logger.debug('emiting data %s -> %s.%s' % (dname, comp_name, f))
                if f not in buff[comp_name]:
                    raise ValueError('Component "%s" does not have a input named "%s"!' \
                        % (comp_name, f))
                buff[comp_name][f] = d_obj
                emitted = True
            if not emitted:
                logger.warning('data "%s" is not emitted to any component' % dname)
        #logger.debug('buffer after emit: %s' % buff)

    def _graph_comp_to_input(self, graph):
        '''
        create mapping:
            comp_name -> input
        '''
        c2i = {}
        #for f, t, cn in graph._edges_with_attr():
        #    c2i[cn] = { fi: None for fi in f }
        for name, (_, inp, out) in graph._comps.iteritems():
            c2i[name] = { i: None for i in slist.iter_multi(inp) }
        return c2i

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

    def compile_graph(self, graph):
        ''' compile a graph to a sequence of computations
        a computation includes:
            - name
            - computing object
            - input buffer: storing input datasets temporarily.
                should be all available before computation starts
                will be cleaned if computation done
            - output: send the output to all computation units that need it
        '''

        self._graph = graph

        self._report('compile graph %s' % graph.name)
        self._report_leveldown()

        logger.info('build output -> input mapping ...')
        oimap = self._build_oimap(graph)

        logger.info('build task sequence ...')
        task_seq = self._build_task_seq(graph)
        self._task_seq = task_seq
        self._report('task sequence:\n%s' % util.indent(self._str_task_seq(), 8))

        self._report(['OI mapping:', \
                ['%d. %s: %s' % (i, k, oimap[k]) \
                    for i, (k, _) in enumerate(task_seq) \
                    if k in oimap]])

        logger.info('replace component names with task index')
        oimap_indexed = self._indexing_comp(oimap, task_seq)
        self._report(['OI map indexed:', \
                ['%d: %s' % (i, oi) for i, oi in enumerate(oimap_indexed)]])
        self._oimap = oimap_indexed

        self._report_levelup()

    def _build_task_seq(self, graph):
        task_seq = [(Executor.__INPUT__, None)]
        def _add_task(entry):
            task_seq.append(entry)
        self.dfs(graph, _add_task)
        task_seq.append((Executor.__OUTPUT__, None))
        return task_seq

    def _indexing_comp(self, oimap, task_seq):
        c2i = { name:i for i, (name, _) in enumerate(task_seq) }
        #print c2i
        oimap_tmp = util.replace_struct(oimap, c2i)
        #print oimap_tmp
        oimap_indexed = [ oimap_tmp[i] for i in sorted(c2i.values()) \
                if i != c2i[Executor.__OUTPUT__]]
        #task_seq_indexed = util.replace_struct(task_seq, c2i)
        return oimap_indexed  #, task_seq_indexed

    def _build_oimap(self, graph):
        oimap = {}
        def _get_oimap_for_outs(cname, outs):
            #one_map = collections.defaultdict(list)
            one_map = []
            logger.debug('build for outputs %s: %s' % (cname, outs))
            for out_idx, out in slist.enumerate_multi(outs):
                logger.debug('+ build for output%s "%s"' % (slist.str_index(out_idx), out))
                for f, t, c_succ in graph._edges_with_attr(out):
                    logger.debug('++ edge: %s -(%s)-> %s' % (f, c_succ, t))
                    inp_idx = slist.index(graph._comps[c_succ].inp, out)
                    logger.debug('++ mapping "%s": %s.out%s -> %s.in%s' \
                            % (out, cname, slist.str_index(out_idx), c_succ, slist.str_index(inp_idx)))
                    one_map.append((out_idx, c_succ, inp_idx))
                if out in slist.iter_multi(graph._out):
                    inp_idx = slist.index(graph._out, out)
                    logger.debug('++ mapping "%s": %s.out%s -> %s.in%s' \
                            % (out, cname, slist.str_index(out_idx), Executor.__OUTPUT__, slist.str_index(inp_idx)))
                    one_map.append((out_idx, Executor.__OUTPUT__, inp_idx))
            return one_map

        #logger.debug('build oimap for input')
        oimap[Executor.__INPUT__] = _get_oimap_for_outs(Executor.__INPUT__, graph._inp)
        logger.debug('build result: %s' % oimap[Executor.__INPUT__])
        for comp_name, entry in graph._comps.iteritems():
            #logger.debug('build out->in mapping for component "%s"' % comp_name)
            curr_map = _get_oimap_for_outs(comp_name, entry.out)
            oimap[comp_name] = curr_map
            logger.debug('build result: %s' % curr_map)
        return oimap

    def run_compiled(self, data, func):
        self._report('Run Compiled Graph "%s" ...' % self._graph.name)

        if self._task_seq is None:
            raise ValueError('No compiled graph found')

        # create buffers for storing real inputs of each task
        logger.debug('creating input buffer ...')
        input_buff = self._create_input_buff()
        #print 'input buff', input_buff

        pending = collections.deque(i for i, _ in enumerate(self._task_seq))

        # run input
        curr_task_no = pending.popleft()
        c_name, entry = self._task_seq[curr_task_no]
        if c_name != Executor.__INPUT__:
            raise ValueError('First task should be input!')
        self._report('Task %d Input: assign input data' % curr_task_no)
        self._report_leveldown()
        self._emit_data_by_index(data, curr_task_no, input_buff)
        self._report_levelup()

        while pending:
            curr_task_no = pending.popleft()
            c_name, c_entry = self._task_seq[curr_task_no]
            curr_input = input_buff[curr_task_no]
            if not self._is_input_ready(curr_input):
                raise ValueError('Task %s does not recieve all input data' % c_name)
            if c_name == Executor.__OUTPUT__:
                self._report('Task %d Ouput: assign output results' % curr_task_no)
                self._report_leveldown()
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
            for cname, (c_obj, c_inp, c_out) in self._task_seq[1:-1]:
                yield c_inp
            yield self._graph._out
        buff = [ slist.nones_like(inp) for inp in _iter_task_inp() ]
        return buff

    def _is_input_ready(self, buff):
        #print '-----> _is_input_ready:', buff
        return all([ d is not None for d in slist.iter_multi(buff)])

    def _emit_data_by_index(self, data, task_no, input_buff):
        def _format_comp(c):
            if c is None:
                c_name = ''
            else:
                c_name = self._task_seq[c][0]
            return c_name

        def _format_output(c, i):
            c_name = self._task_seq[c][0]
            if c_name == Executor.__INPUT__:
                i_name = slist.get(self._graph._inp, i)
            else:
                i_name = slist.get(self._task_seq[c][1].out, i)
            return 'output%s:"%s"' % (slist.str_index(i), i_name)

        def _format_input(c, i):
            c_name = self._task_seq[c][0]
            if c_name == Executor.__OUTPUT__:
                i_name = slist.get(self._graph._out, i)
            else:
                i_name = slist.get(self._task_seq[c][1].inp, i)
            return 'input%s:"%s"' % (slist.str_index(i), i_name)

        oimap = self._oimap[task_no]
        for o, c, i in oimap:
            self._report('emit data "%s".%s -> "%s".%s' \
                    % (_format_comp(task_no), _format_output(task_no, o), \
                       _format_comp(c), _format_input(c, i)))
            di = slist.get(data, o)
            if i is None:
                input_buff[c]  = di
            else:
                input_buff[c][i] = di

    def _str_task_seq(self):
        def _str_task(i, x):
            name, entry = x
            if name == Executor.__INPUT__:
                s = 'INPUT: {}'.format(str(self._graph._inp))
            elif name == Executor.__OUTPUT__:
                s = 'OUTPUT: {}'.format(str(self._graph._out))
            else:
                s = '{}\n{}'.format(name, util.indent(str(entry)))
            return '%d. %s' % (i, s)
        jnr = '\n'
        return jnr.join(_str_task(i, x) for i, x in enumerate(self._task_seq))

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

    exe = Executor(report.TxtReporter())
    data = [(11,22), {'a':100, 10:200}]
    #exe.run_train(data, gcomp)
    exe.compile_graph(gcomp)
    exe.run_train(data)

def main():
    #test_maybe_list()
    test_exec()
    return

if __name__ == '__main__':
    main()
