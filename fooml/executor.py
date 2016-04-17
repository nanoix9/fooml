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
            out = self._train_graph(acomp, data)
        else:
            #print acomp
            #self._report('training basic "%s" ...' % acomp.name)
            out = self._train_one(acomp, data)
        return out

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
        print '--> emit data:', data, data_names, graph
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
                out_buff[dname] = d_obj;
            for f, t, comp_name in graph._edges_with_attr(nbunch=[dname]):
                print '__emit_data', f, t, comp_name
                if f not in buff[comp_name]:
                    raise ValueError('Component "%s" does not have a input named "%s"!' \
                        % (comp_name, f))
                buff[comp_name][f] = d_obj
        #print(buff)

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

def iter_maybe_list(obj, *args):
    #print 'iter_maybe_list args:', obj, args
    if isinstance(obj, (tuple, list)):
        if any([len(a) != len(obj) for a in args]):
            raise ValueError('length of lists are not identical')
        for i, o in enumerate(obj):
            yield tuple_or_scale([o] + [a[i] for a in args])
    else:
        yield tuple_or_scale((obj,) + args)

def tuple_or_scale(obj):
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



######## tests

def test_maybe_list():
    print [ a for a in iter_maybe_list('a')]
    print [ a for a in iter_maybe_list(['a', 1, 2])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20, 30])]
    print [ a for a in iter_maybe_list(['a', 1], [10, 20], [100, 110])]
    print [ a for a in iter_maybe_list(['a', 1, 2], [10, 20])]

def test_exec():
    gcomp = comp.GraphComp('test_graph', inp=['input', 'x'], out='y')
    gcomp.add_comp('c1', comp.PassComp(), 'x', 'u')
    gcomp.add_comp('c2', comp.ConstComp(1), ['input', 'u'], 'y')
    print gcomp

    import report
    exe = Executor(report.TxtReporter())
    exe.run_train(gcomp, [(11,22), {'a':100, 10:200}])

def main():
    #test_maybe_list()
    test_exec()
    return

if __name__ == '__main__':
    main()
