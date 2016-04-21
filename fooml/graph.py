#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import networkx as nx
import collections as c
import comp
import util


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
        self._graph.add_edges_from(conn, **attr)

    def add_comp(self, name, acomp, inp, out):
        if name in self._comps:
            raise ValueError('Component %s already exists!' %s)
        self._comps[name] = _entry(acomp, inp, out)
        inp_list = util.to_list(inp)
        out_list = util.to_list(out)
        self._add_nodes(inp_list)
        self._add_nodes(out_list)
        self._add_edges(inp_list, out_list, name=name, comp=acomp)
        return self

    def __str__(self):
        return '%s:\n\t Input:  %s,\n\t Output: %s,\n\t Nodes:  %s,\n\t Edges:  %s,\n\t Components:\n%s]' % \
                (self.__class__.__name__, self._inp, self._out, \
                 self._graph.nodes(), \
                 self._str_edges_with_attr(), \
                 self._str_comps())

    def _str_edges_with_attr(self, attr='name'):
        return str(self._edges_with_attr(attr=attr))

    def _edges_with_attr(self, nbunch=None, attr='name'):
        if isinstance(attr, (list, tuple)):
            get = lambda f, t, attr: tuple([self._graph[f][t][a] for a in attr])
        else:
            get = lambda f, t, attr: (self._graph[f][t][attr],)
        ls = [(f, t) + get(f, t, attr) for f, t in self._graph.edges_iter(nbunch=nbunch)]
        return ls

    def _str_comps(self):
        slist = []
        for name, c in self._comps.iteritems():
            slist.append('\'%s\':\n%s' % (name, util.indent(str(c))))
        return '\n'.join(slist)



def test_graph():

    gcomp = CompGraph('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', None, 'x', 'u')
    gcomp.add_comp('c2', None, ['input', 'u'], 'y')
    print gcomp

def main():
    test_graph()
    return

if __name__ == '__main__':
    main()
