#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function

import sys
import comp
import collections


class Executor(object):

    __INPUT__ = '__INPUT__'
    __OUTPUT__ = '__OUTPUT__'

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

    def run_train(self, start_data, acomp=None):
        data = start_data
        self._report_leveldown()
        if acomp is None:
            self.run_compiled(data)
        else:
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
        elif isinstance(acomp, comp.GraphComp):
            self._report('training graph "%s" ...' % acomp.name)
            out = self.run_train(acomp, data)
        else:
            #print acomp
            #self._report('training basic "%s" ...' % acomp.name)
            out = self._train_one(acomp, data)
        return out

    def dfs(self, graph, func):
        data = ones_like(graph._inp)
        buff = self._graph_comp_to_input(graph)
        out_buff = {o: None for o in iter_maybe_list(graph._out)}
        print buff
        self.__emit_data(data, graph._inp, graph, buff, out_buff)
        stack = _to_list(graph._inp, copy=True)
        visited = set()
        while stack:
            curr_node = stack.pop()
            self._report('DFS visits node "%s"' \
                    % (curr_node))
            if curr_node in visited:
                raise ValueError('Graph has cycle(s) in it' % graph.name)
            visited.add(curr_node)
            for f, t, comp_name, acomp in graph._edges_with_attr(curr_node, attr=('name', 'comp')):
                print '>>> DFS checking edge:', f, t, comp_name, acomp
                curr_input = buff[comp_name]
                entry = graph._comps[comp_name]
                if self.__is_inputs_ready(curr_input):
                    self._report('DFS got component "%s" ready for processing ...' \
                            % (comp_name,))
                    #real_input_data = self.__make_real_input(curr_input, entry.inp)
                    #print '>>> train:', acomp, real_input_data
                    #out = self._train_component(acomp, real_input_data)
                    #print '>>> train out:', out
                    func((comp_name, entry))
                    out = ones_like(entry.out)
                    self.__clear_inputs(curr_input)
                    self.__emit_data(out, entry.out, graph, buff, out_buff)
                    stack.extend(_to_list(entry.out))
                    print '>>> DFS current output buffer of graph:', out_buff
        if any(d is None for n, d in out_buff.iteritems()):
            raise ValueError('Nothing is connected to output(s): %s' \
                % filter(lambda n: out_buff[n] is None, out_buff.keys()))
        ret = gets_from_dict(out_buff, graph._out)
        print('DFS graph final output: %s' % ret)
        return ret

    def _train_graph(self, graph, data):
        buff = self._graph_comp_to_input(graph)
        out_buff = {o: None for o in iter_maybe_list(graph._out)}
        print buff
        self.__emit_data(data, graph._inp, graph, buff, out_buff)
        stack = _to_list(graph._inp, copy=True)
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
                    real_input_data = self.__make_real_input(curr_input, entry.inp)
                    print '>>> train:', acomp, real_input_data
                    out = self._train_component(acomp, real_input_data)
                    print '>>> train out:', out
                    self.__clear_inputs(curr_input)
                    #out_names = _to_list(entry.out)
                    self.__emit_data(out, entry.out, graph, buff, out_buff)
                    stack.extend(_to_list(entry.out))
                    print '>>> out of graph:', out_buff
        if any(d is None for n, d in out_buff.iteritems()):
            raise ValueError('Output did not get an value: %s' \
                % filter(lambda n: out_buff[n] is None, out_buff.keys()))
        ret = gets_from_dict(out_buff, graph._out)
        print('final output: %s' % ret)
        return ret

    def __is_inputs_ready(self, buff):
        return all([d is not None for i, d in buff.iteritems()])

    def __make_real_input(self, data_dict, input_names):
        return map_maybe_list(lambda n: data_dict[n], input_names)

    def __clear_inputs(self, buff):
        for k in buff:
            buff[k] = None

    def __emit_data(self, data, data_names, graph, buff, out_buff):
        print '--> emit data "%s": %s' % (data, data_names)
        if any(d is None for d in iter_maybe_list(data)):
            raise ValueError('real data is none for data with name "%s"' % data_names)
        data_dict = { n:d for n, d in iter_maybe_list(data_names, data) }
        #print data_dict
        #sys.exit()
        print('before emit:', buff)
        for dname, d_obj in data_dict.iteritems():
            if dname in out_buff:
                if out_buff[dname] is not None:
                    raise ValueError('output of "%s" already got a value' % dname)
                print '>>>> emit to output "%s": %s' % (dname, d_obj)
                out_buff[dname] = d_obj;
            for f, t, comp_name in graph._edges_with_attr(nbunch=[dname]):
                print '>>>> emiting data "%s" to %s->%s' % (dname, comp_name, f)
                if f not in buff[comp_name]:
                    raise ValueError('Component "%s" does not have a input named "%s"!' \
                        % (comp_name, f))
                buff[comp_name][f] = d_obj
        print('after emit:', buff)

    def _graph_comp_to_input(self, graph):
        '''
        create mapping:
            comp_name -> input
        '''
        c2i = {}
        #for f, t, cn in graph._edges_with_attr():
        #    c2i[cn] = { fi: None for fi in f }
        for name, (_, inp, out) in graph._comps.iteritems():
            c2i[name] = { i: None for i in inp }
        return c2i

    def _train_one(self, basic_comp, data):
        return basic_comp.fit_trans(data)

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
        def pt(x):
            print x
        oimap = self._build_oimap(graph)
        task_seq = self._build_task_seq(graph)
        print 'OI mapping:', oimap
        print 'task sequence:', task_seq
        oimap_indexed = self._indexing_comp(oimap, task_seq)
        print 'OI map indexed:', oimap_indexed

        self._oimap = oimap_indexed
        self._task_seq = task_seq
        self._graph = graph

    def _build_task_seq(self, graph):
        task_seq = [(Executor.__INPUT__, None)]
        def _add_task(entry):
            task_seq.append(entry)
        self.dfs(graph, _add_task)
        task_seq.append((Executor.__OUTPUT__, None))
        return task_seq

    def _indexing_comp(self, oimap, task_seq):
        c2i = { name:i for i, (name, _) in enumerate(task_seq) }
        print '-----_indexing_comp---------'
        #print c2i
        oimap_tmp = replace_struct(oimap, c2i)
        print oimap_tmp
        oimap_indexed = [ oimap_tmp[i] for i in sorted(c2i.values()) \
                if i != c2i[Executor.__OUTPUT__]]
        #task_seq_indexed = replace_struct(task_seq, c2i)
        return oimap_indexed  #, task_seq_indexed

    def _build_oimap(self, graph):
        oimap = {}
        def _get_oimap_for_outs(cname, outs):
            #one_map = collections.defaultdict(list)
            one_map = []
            print '>>>> build oimap for outs of component "%s": %s' % (cname, outs)
            for out_idx, out in enumerate_maybe_list(outs):
                print '>>>> build oimap for out %s: "%s"' % (out_idx, out)
                for f, t, c_succ in graph._edges_with_attr(out):
                    print '>>>> edge: %s->%s:%s' % (f, t, c_succ)
                    inp_idx = call_maybe_list(graph._comps[c_succ].inp, list.index, out)
                    print '>>>> mapping "%s": %s.out%s -> %s.inp%s' \
                            % (out, cname, _str_index(out_idx), c_succ, _str_index(inp_idx))
                    one_map.append((out_idx, c_succ, inp_idx))
                if out in iter_maybe_list(graph._out):
                    inp_idx = call_maybe_list(graph._out, list.index, out)
                    one_map.append((out_idx, Executor.__OUTPUT__, inp_idx))
            return one_map

        oimap[Executor.__INPUT__] = _get_oimap_for_outs(Executor.__INPUT__, graph._inp)
        print '>>>> build oimap for input',  oimap
        for comp_name, entry in graph._comps.iteritems():
            print '>>>> build oimap for component "%s"' % comp_name
            curr_map = _get_oimap_for_outs(comp_name, entry.out)
            oimap[comp_name] = curr_map
            print '>>>> after build:', (oimap)
        return oimap

    def run_compiled(self, data):
        self._report('Run Compiled Graph "%s" ...' % self._graph.name)

        if self._task_seq is None:
            raise ValueError('No compiled graph found')

        # create buffers for storing real inputs of each task
        input_buff = self._create_input_buff()
        print 'input buff', input_buff

        pending = collections.deque(i for i, _ in enumerate(self._task_seq))

        # run input
        curr_task_no = pending.popleft()
        c_name, entry = self._task_seq[curr_task_no]
        if c_name != Executor.__INPUT__:
            raise ValueError('First task should be input!')
        self._report('Task Input: assign input data')
        self._report_leveldown()
        self.__emit_data_by_index(data, curr_task_no, input_buff)
        self._report_levelup()

        while pending:
            curr_task_no = pending.popleft()
            c_name, c_entry = self._task_seq[curr_task_no]
            curr_input = input_buff[curr_task_no]
            if not self._is_input_ready(curr_input):
                raise ValueError('Task %s does not recieve all input data' % c_name)
            if c_name == Executor.__OUTPUT__:
                self._report('Task Ouput: assign output results')
                self._report_leveldown()
                ret = curr_input
                self._report_levelup()
            else:
                c_obj, c_inp, c_out = c_entry
                self._report('Task %d: train component "%s"%s, input=%s, output=%s' \
                    % (curr_task_no, c_name, c_obj.__class__, c_inp, c_out))
                self._report_leveldown()
                out = self._train_component(c_obj, curr_input)
                self.__emit_data_by_index(out, curr_task_no, input_buff)
                input_buff[curr_task_no] = None  # clean input data
                self._report_levelup()
        return ret

    def _create_input_buff(self):
        def _iter_task_inp():
            yield self._graph._inp
            for cname, (c_obj, c_inp, c_out) in self._task_seq[1:-1]:
                yield c_inp
            yield self._graph._out
        buff = [ nones_like(inp) for inp in _iter_task_inp() ]
        return buff

    def _is_input_ready(self, buff):
        #print '-----> _is_input_ready:', buff
        return all([ d is not None for d in iter_maybe_list(buff)])

    def __emit_data_by_index(self, data, task_no, input_buff):
        def _format_comp(c):
            if c is None:
                c_name = ''
            else:
                c_name = self._task_seq[c][0]
            return c_name

        def _format_output(c, i):
            c_name = self._task_seq[c][0]
            if c_name == Executor.__INPUT__:
                i_name = get_maybe_list(self._graph._inp, i)
            else:
                i_name = get_maybe_list(self._task_seq[c][1].out, i)
            return 'output%s:"%s"' % (_str_index(i), i_name)

        def _format_input(c, i):
            c_name = self._task_seq[c][0]
            if c_name == Executor.__OUTPUT__:
                i_name = get_maybe_list(self._graph._out, i)
            else:
                i_name = get_maybe_list(self._task_seq[c][1].inp, i)
            return 'input%s:"%s"' % (_str_index(i), i_name)

        oimap = self._oimap[task_no]
        for o, c, i in oimap:
            self._report('emit "%s".%s --> "%s".%s' \
                    % (_format_comp(task_no), _format_output(task_no, o), \
                       _format_comp(c), _format_input(c, i)))
            di = get_maybe_list(data, o)
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


def _to_list(obj, copy=False):
    if isinstance(obj, (tuple, list)):
        if copy:
            return list(obj)
        else:
            return obj
    else:
        return [obj]

def call_maybe_list(obj, func, *args):
    if isinstance(obj, (tuple, list)):
        return func(obj, *args)
    else:
        return None

def _str_index(idx):
    if idx is None:
        return ''
    else:
        return '[%s]' % idx

def nones_like(obj):
    return rep_like_maybe_list(None, obj)

def ones_like(obj):
    return rep_like_maybe_list(1, obj)

def rep_like_maybe_list(v, obj):
    return map_maybe_list(lambda x: v, obj)

def get_maybe_list(obj, idx):
    if isinstance(obj, (tuple, list)):
        return obj[idx]
    else:
        if idx is not None:
            raise ValueError('index is not None but object is not a list')
        return obj

def enumerate_maybe_list(obj, *args):
    #print 'iter_maybe_list args:', obj, args
    if isinstance(obj, (tuple, list)):
        if any([len(a) != len(obj) for a in args]):
            raise ValueError('length of lists are not identical')
        for i, o in enumerate(obj):
            yield maybe_tuple([i, o] + [a[i] for a in args])
    else:
        yield maybe_tuple((None, obj) + args)

def iter_maybe_list(obj, *args):
    #print 'iter_maybe_list args:', obj, args
    if isinstance(obj, (tuple, list)):
        if any([len(a) != len(obj) for a in args]):
            raise ValueError('length of lists are not identical')
        for i, o in enumerate(obj):
            yield maybe_tuple([o] + [a[i] for a in args])
    else:
        yield maybe_tuple((obj,) + args)

def maybe_tuple(obj):
    if isinstance(obj, (tuple, list)):
        if len(obj) > 1:
            return tuple(obj)
        else:
            return obj[0]
    else:
        return obj

def map_maybe_list(func, obj):
    if isinstance(obj, (tuple, list)):
        return map(func, obj)
    else:
        return func(obj)

def gets_from_dict(adict, keys):
    ''' return a list while `keys` is a list or tuple of keys;
    or a value if `keys` is a single key
    '''

    if isinstance(keys, (tuple, list)):
        ret = [ adict[k] for k in keys ]
        if isinstance(keys, tuple):
            return tuple(ret)
        else:
            return ret
    else:
        return adict[keys]

def replace_struct(obj, replace):
    if isinstance(obj, dict):
        return {replace_struct(k, replace): replace_struct(v, replace) \
                for k, v in obj.iteritems()}
    elif isinstance(obj, list):
        return [replace_struct(i, replace) for i in obj]
    elif isinstance(obj, tuple):
        return tuple([replace_struct(i, replace) for i in obj])
    else:
        return replace.get(obj, obj)


######## tests

def test_maybe_list():
    print [ a for a in iter_maybe_list('a')]
    print [ a for a in iter_maybe_list(['a', 1, 2])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20, 30])]
    print [ a for a in iter_maybe_list(['a', 1], [10, 20], [100, 110])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20])]

def test_replace_struct():
    s = {'a':[('a', 1), ['ab', 'a'], 'a', 100], 2:'a'}
    rep = {'a': 'x'}
    r = replace_struct(s, rep)
    print s
    print r

def test_exec():
    gcomp = comp.GraphComp('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', comp.PassComp(), 'x', 'u')
    gcomp.add_comp('c2', comp.ConstComp(1), ['input', 'u'], 'z')
    #gsub1 = comp.GraphComp('subgraph1', inp='s1', out='y1')
    #gsub1.add_comp('c31', comp.PassComp(), 's1', 'y1')
    #gcomp.add_comp('g3', gsub1, 'z', 'y')
    gcomp.add_comp('g3', comp.PassComp(), 'z', 'y')
    print gcomp

    import report
    exe = Executor(report.TxtReporter())
    data = [(11,22), {'a':100, 10:200}]
    #exe.run_train(data, gcomp)
    exe.compile_graph(gcomp)
    exe.run_train(data)

def main():
    #test_maybe_list()
    test_replace_struct()
    test_exec()
    return

if __name__ == '__main__':
    main()
