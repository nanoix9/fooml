#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import networkx as nx
import collections as c
import comp
from dt import slist
import util
from log import logger

__INPUT__ = '__INPUT__'
__OUTPUT__ = '__OUTPUT__'
__NULL__ = '_'

_entry = c.namedtuple('_entry', 'comp, inp, out')
def _str_entry(entry):
    return 'input: {input}\noutput: {output}\ncomponent: {comp}' \
            .format(input=entry.inp, output=entry.out, comp=entry.comp)
_entry.__str__ = _str_entry
_entry.__repr__ = _str_entry

class CompGraph(object):

    def __init__(self, name, inp=None, out=None):
        self.name = name
        self._graph = nx.DiGraph()
        self._inp = None
        self._out = None
        self.set_input(inp)
        self.set_output(out)
        self._comps = {}

    def set_input(self, inp):
        if inp is None:
            return
        self._inp = inp
        self._add_nodes(inp)

    def set_output(self, out):
        if out is None:
            return
        self._out = out
        self._add_nodes(out)

    def _add_nodes(self, nodes):
        if isinstance(nodes, (int, str, unicode)):
            self._graph.add_node(nodes)
        else:
            self._graph.add_nodes_from(nodes)

    def _add_edges(self, inp, out, **attr):
        def _iter(x):
            if isinstance(x, (int, str, unicode)):
                yield x
            else:
                for i in x:
                    yield x
        conn = [ (x, y) for x in inp for y in out]
        for x, y in conn:
            if self._graph.has_edge(x, y):
                raise RuntimeError('already an edge between %d and %d' % (x, y))
        self._graph.add_edges_from(conn, **attr)

    def add_comp(self, name, acomp, inp, out):
        if name in self._comps:
            raise ValueError('Component %s already exists!' %s)
        self._comps[name] = _entry(acomp, inp, out)
        inp_list = slist.to_list(inp)
        out_list = slist.to_list(out)
        self._add_nodes(inp_list)
        self._add_nodes(out_list)
        self._add_edges(inp_list, out_list, name=name, comp=acomp)
        return self

    def get_comp_entry(self, name):
        return self._comps[name]

    def get_comp(self, name):
        return self.get_comp_entry(name).comp

    def iter_comps(self):
        for name, entry in self._comps.iteritems():
            yield name, entry.comp

    def __str__(self):
        return '%s: %s\n  Input:  %s,\n  Output: %s,\n  Nodes:  %s,\n  Edges:\n%s,\n  Components:\n%s' % \
                (self.__class__.__name__, self.name, self._inp, self._out, \
                 self._graph.nodes(), \
                 util.indent(self._str_edges_with_attr(), 4), \
                 util.indent(self._str_comps(), 2, '  '))

    def _str_edges_with_attr(self, attr='name'):
        return util.joins(self._pretty_edge(e) for e in self._edges_with_attr(attr=attr))

    def _edges_with_attr(self, nbunch=None, attr='name'):
        if isinstance(attr, (list, tuple)):
            get = lambda f, t, attr: tuple([self._graph[f][t][a] for a in attr])
        else:
            get = lambda f, t, attr: (self._graph[f][t][attr],)
        ls = [(f, t) + get(f, t, attr) for f, t in self._graph.edges_iter(nbunch=nbunch)]
        return ls

    def _pretty_edge(self, e):
        return '%s --> %s: %s' % (e[0], e[1], str(e[2:]))

    def _str_comps(self):
        slist = []
        for name, c in self._comps.iteritems():
            slist.append('\'%s\':\n%s' % (name, util.indent(str(c))))
        return '\n'.join(slist)

    def dfs(self, func):
        logger.info('start Depth-First Searching of graph %s ...' % self.name)

        logger.debug('build fake input data to mark node visiting')
        data = slist.ones_like(self._inp)
        logger.debug('fake input data: %s' % str(data))

        logger.debug('build input buffer for each edge and the final output')
        buff = self._graph_comp_to_input()
        #print buff
        out_buff = {o: None for o in slist.iter_multi(self._out)}

        logger.debug('setup input data to initialize graph searching')
        self._emit_data(data, self._inp, buff, out_buff)
        stack = slist.to_list(self._inp, copy=True)
        visited = set()
        while stack:
            curr_node = stack.pop()
            logger.debug('visits node "%s"' % (curr_node))
            if curr_node in visited and curr_node != __NULL__:
                raise ValueError('node "%s" has already been visited' % curr_node)
            visited.add(curr_node)
            for f, t, comp_name, acomp in self._edges_with_attr(curr_node, attr=('name', 'comp')):
                logger.debug('+ checking edge: %s -(%s)-> %s' % (f, comp_name, t)) #, acomp)
                curr_input = buff[comp_name]
                entry = self._comps[comp_name]
                if self.__is_inputs_ready(curr_input):
                    logger.debug('+ edge "%s" is ready for visiting' % (comp_name,))
                    func((comp_name, entry))
                    out = slist.ones_like(entry.out)  # fake output data
                    self.__clear_inputs(curr_input)
                    self._emit_data(out, entry.out, buff, out_buff)
                    stack.extend(slist.to_list(entry.out))
                    logger.debug('+ current output buffer of graph: %s' % out_buff)
        if any(d is None for n, d in out_buff.iteritems()):
            raise ValueError('Nothing is connected to output(s): %s' \
                % filter(lambda n: out_buff[n] is None, out_buff.keys()))
        ret = util.gets_from_dict(out_buff, self._out)
        logger.debug('graph final output: %s' % ret)
        return ret

    def _graph_comp_to_input(self):
        '''
        create mapping:
            comp_name -> input
        '''
        c2i = {}
        #for f, t, cn in self._edges_with_attr():
        #    c2i[cn] = { fi: None for fi in f }
        for name, (_, inp, out) in self._comps.iteritems():
            c2i[name] = { i: None for i in slist.iter_multi(inp) }
        return c2i

    def _emit_data(self, data, data_names, buff, out_buff):
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
                if dname != __NULL__ and out_buff[dname] is not None:
                    raise ValueError('output of "%s" already got a value' % dname)
                logger.debug('emit to output "%s": %s' % (dname, d_obj))
                out_buff[dname] = d_obj;
                emitted = True
            for f, t, comp_name in self._edges_with_attr(nbunch=[dname]):
                logger.debug('emiting data %s -> %s.%s' % (dname, comp_name, f))
                if f not in buff[comp_name]:
                    raise ValueError('Component "%s" does not have a input named "%s"!' \
                        % (comp_name, f))
                buff[comp_name][f] = d_obj
                emitted = True
            if not emitted:
                logger.warning('data "%s" is not emitted to any component' % dname)

    def __is_inputs_ready(self, buff):
        return all([d is not None for i, d in buff.iteritems()])

    def __clear_inputs(self, buff):
        for k in buff:
            buff[k] = None


class _CompiledGraph(object):

    def __init__(self, graph=None):
        if graph is not None:
            self.compile_graph(graph)

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

        logger.info('compile graph %s' % graph.name)

        logger.info('build output -> input mapping ...')
        oimap = self._build_oimap(graph)
        self._oimap_named = oimap

        logger.info('build task sequence ...')
        task_seq = self._build_task_seq(graph)
        self._task_seq = task_seq

        logger.info('replace component names with task index')
        oimap_indexed = self._indexing_comp(oimap, task_seq)
        self._oimap = oimap_indexed

    def _build_task_seq(self, graph):
        task_seq = [(__INPUT__, None)]
        def _add_task(entry):
            task_seq.append(entry)
        graph.dfs(_add_task)
        task_seq.append((__OUTPUT__, None))
        return task_seq

    def _indexing_comp(self, oimap, task_seq):
        c2i = { name:i for i, (name, _) in enumerate(task_seq) }
        #print c2i
        oimap_tmp = util.replace_struct(oimap, c2i)
        #print oimap_tmp
        oimap_indexed = [ oimap_tmp[i] for i in sorted(c2i.values()) \
                if i != c2i[__OUTPUT__]]
        #task_seq_indexed = util.replace_struct(task_seq, c2i)
        return oimap_indexed  #, task_seq_indexed

    def _build_oimap(self, graph):
        '''convert:
            comp_prev.output{'name'} --> comp_next.input{'name'}
        to
            comp_prev.output[i] --> comp_next.input[j]
        '''

        oimap = {}
        #logger.debug('build oimap for input')
        oimap[__INPUT__] = self._get_oimap_for_outs(graph, __INPUT__, graph._inp)
        logger.debug('build result: %s' % oimap[__INPUT__])
        for comp_name, entry in graph._comps.iteritems():
            #logger.debug('build out->in mapping for component "%s"' % comp_name)
            curr_map = self._get_oimap_for_outs(graph, comp_name, entry.out)
            oimap[comp_name] = curr_map
            logger.debug('build result: %s' % curr_map)
        return oimap

    def _get_oimap_for_outs(self, graph, cname, outs):
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
                        % (out, cname, slist.str_index(out_idx), __OUTPUT__, slist.str_index(inp_idx)))
                one_map.append((out_idx, __OUTPUT__, inp_idx))
        return one_map

    def str_task_seq(self):
        def _str_task(i, x):
            name, entry = x
            if name == __INPUT__:
                s = 'INPUT: {}'.format(str(self._graph._inp))
            elif name == __OUTPUT__:
                s = 'OUTPUT: {}'.format(str(self._graph._out))
            else:
                s = '{}\n{}'.format(name, util.indent(str(entry)))
            return '%d. %s' % (i, s)
        jnr = '\n'
        return jnr.join(_str_task(i, x) for i, x in enumerate(self._task_seq))

    def format_comp(self, c):
        if c is None:
            c_name = ''
        else:
            c_name = self._task_seq[c][0]
        return c_name

    def format_output(self, c, i):
        c_name = self._task_seq[c][0]
        if c_name == __INPUT__:
            i_name = slist.get(self._graph._inp, i)
        else:
            i_name = slist.get(self._task_seq[c][1].out, i)
        return 'output%s:"%s"' % (slist.str_index(i), i_name)

    def format_input(self, c, i):
        c_name = self._task_seq[c][0]
        if c_name == __OUTPUT__:
            i_name = slist.get(self._graph._out, i)
        else:
            i_name = slist.get(self._task_seq[c][1].inp, i)
        return 'input%s:"%s"' % (slist.str_index(i), i_name)

    def __str__(self):

        return util.joins([
                '%s: %s' % (self.__class__.__name__, self._graph.name),
                'task sequence:',
                util.indent(self.str_task_seq(), 2),

                'OI mapping:',
                ['%d. %s: %s' % (i, k, self._oimap_named[k]) \
                    for i, (k, _) in enumerate(self._task_seq) \
                    if k in self._oimap_named],

                'OI map indexed:',
                ['%d: %s' % (i, oi) for i, oi in enumerate(self._oimap)]
            ])



def test_graph():

    gcomp = CompGraph('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', None, 'x', 'u')
    gcomp.add_comp('c2', None, ['input', 'u'], 'y')
    print gcomp

def test_compile():
    gcomp = CompGraph('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', comp.PassComp(), 'x', 'u')
    gcomp.add_comp('c2', comp.ConstComp(1), ['input', 'u'], 'z')
    #gsub1 = graph.CompGraph('subgraph1', inp='s1', out='y1')
    #gsub1.add_comp('c31', comp.PassComp(), 's1', 'y1')
    #gcomp.add_comp('g3', gsub1, 'z', 'y')
    gcomp.add_comp('g3', comp.PassComp(), 'z', 'y')
    print gcomp

    cg = _CompiledGraph()
    cg.compile_graph(gcomp)


def main():
    test_graph()
    test_compile()
    return

if __name__ == '__main__':
    main()
