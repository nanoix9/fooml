#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp
import collections as c


class Executor(object):

    def __init__(self, reporter):
        self._reporter = reporter
        #self._comp = comp.Serial()

    def add_component(self, obj, name=None, func=None):
        if not isinstance(obj, comp.Component):
            c = comp.Component(name, obj)
        else:
            c = obj
        self._comp.add_component(c)

    def run_train(self, acomp, start_data):
        data = start_data
        self._report_leveldown()
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
            self._train_graph(acomp, data)
        else:
            #print acomp
            #self._report('training basic "%s" ...' % acomp.name)
            d = self._train_one(acomp, data)

    def _train_graph(self, graph, data):
        buff = self._graph_comp_to_input(graph)
        print buff
        self.__emit_data(data, _to_list(graph._inp), graph, buff)
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
                    inp_data_list = [ curr_input[i] for i in entry.inp ]
                    out = self._train_component(acomp, inp_data_list)
                    self.__clear_inputs(curr_input)
                    out_names = _to_list(entry.out)
                    self.__emit_data(out, out_names, graph, buff)
                    for o in out_names:
                        stack.append(o)

    def __is_inputs_ready(self, buff):
        return all([d is not None for i, d in buff.iteritems()])

    def __clear_inputs(self, buff):
        for k in buff:
            buff[k] = None

    def __emit_data(self, data, data_names, graph, buff):
        print '-->', data, data_names, graph
        if len(data) != len(data_names):
            raise ValueError('number of objects in real data does not match data names')
        data_dict = { o: data[i] for i, o in enumerate(data_names)}
        print('before emit:', buff)
        for dname, d_obj in data_dict.iteritems():
            for f, t, comp_name in graph._edges_with_attr(nbunch=[dname]):
                print '__emit_data', f, t, comp_name
                if f not in buff[comp_name]:
                    raise ValueError('Component "%s" does not have a input named "%s"!' \
                        % (comp_name, f))
                buff[comp_name][f] = d_obj
        print(buff)

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

    def _train_one(self, obj, data):
        pass

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

def test_exec():
    gcomp = comp.GraphComp('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', None, 'x', 'u')
    gcomp.add_comp('c2', None, ['input', 'u'], 'y')
    print gcomp

    import report
    exe = Executor(report.TxtReporter())
    exe.run_train(gcomp, [(11,22), {'a':100, 10:200}])

def main():
    test_exec()
    return

if __name__ == '__main__':
    main()
