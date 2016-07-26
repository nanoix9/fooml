#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp


class Group(comp.Comp):
    pass

class ExecMixin(object):

    def set_exec(self, e):
        self._obj = e

    #def set_graph(self, graph):
    #    self._obj.set_graph(graph)

    def _fit_trans_impl(self, data):
        return self._obj.run_train(data)

    def _trans_impl(self, data):
        return self._obj.run_test(data)

class SeqMixin(object):

    def fit_trans(self, data):
        return self._fit_trans_impl(data)

    def trans(self, data):
        return self._trans_impl(data)

class LoopMixin(object):

    def fit_trans(self, data):
        for d in self._loop(data):
            self._fit_trans_impl(d)

    def trans(self, data):
        for d in self._loop(data):
            self._trans_impl(d)

    def _loop(self, data):
        raise NotImplementedError()

class Seq(SeqMixin, ExecMixin, Group):

    def __init__(self, e):
        self.set_exec(e)

class Loop(LoopMixin, ExecMixin, Group):

    def __init__(self, e):
        self.set_exec(e)

def main():
    return

if __name__ == '__main__':
    main()
