#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import networkx as nx
import collections as c


class Component(object):

    def __init__(self, name, obj):
        self.name = name
        self.obj = obj

class _CompList(Component):

    def __init__(self, name):
        super(_CompList, self).__init__(name, [])

    def add_obj(self, name, obj):
        self.obj.append(Component(name, obj))

    def add_component(self, name, comp):
        self.obj.append(comp)

    def __iter__(self):
        for o in self.obj:
            yield o

class Parallel(_CompList):

    def __init__(self, name='parallel'):
        super(Parallel, self).__init__(name)
        #self._objs = (name, [])

class Serial(_CompList):

    def __init__(self, name='serial'):
        super(Serial, self).__init__(name)
        #self._objs = []

class Comp(object):

    def __init__(self, obj):
        self._obj = obj

_entry = c.namedtuple('_entry', 'comp, inp, out')

class GraphComp(object):

    def __init__(self, name, inp, out):
        self._graph = nx.DiGraph()
        self._inp = inp
        self._out = out
        self._add_nodes(inp)
        self._add_nodes(out)
        self._comps = {}

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
        self._add_nodes(inp)
        self._add_nodes(out)
        self._add_edges(inp, out, name=name)
        return self

    def __str__(self):
        return '<%s>[Nodes:%s, Edges:%s, Components:%s]' % \
                (self.__class__.__name__, self._graph.nodes(), self._graph.edges(), \
                self._str_comps())

    def _str_comps(self):
        slist = []
        for name, c in self._comps.iteritems():
            slist.append('\'%s\':%s' % (name, str(c.comp)))
        return '{%s}' % ', '.join(slist)

def test_graph():

    gcomp = GraphComp('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', None, 'x', 'u')
    gcomp.add_comp('c2', None, ['input', 'u'], 'y')
    print gcomp

def main():
    test_graph()
    return

if __name__ == '__main__':
    main()
