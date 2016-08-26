#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import comp

class ExecMixin(object):

    def set_exec(self, e):
        self._exec = e

    #def set_graph(self, graph):
    #    self._exec.set_graph(graph)

    def _fit_trans_impl(self, data):
        return self._exec.run_train(data)

    def _trans_impl(self, data):
        return self._exec.run_test(data)

    def _compile_impl(self, exe):
        self._exec.set_reporter(exe._reporter)
        return self._exec.compile()

    #def __repr__(self):
    #    pass

    def __str__(self):
        return str(self._exec)

class EmbedMixin(object):

    def set_model(self, model):
        if not isinstance(model, comp.Comp):
            raise TypeError()
        self._model = model

    def compile(self, exe):
        if isinstance(self._model, ExecComp):
            self._model.compile(exe)
        else:
            raise NotImplementedError()

class WrapMixin(object):

    def fit_trans(self, data):
        return self._fit_trans_impl(data)

    def trans(self, data):
        return self._trans_impl(data)

    def compile(self, exe):
        return self._compile_impl(exe)

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

class Seq(SeqMixin, ExecMixin, comp.Comp):

    def __init__(self, e):
        super(Seq, self).__init__(e)
        self.set_exec(e)

class ExecComp(ExecMixin, WrapMixin, comp.Comp):

    def __init__(self, exe):
        super(ExecComp, self).__init__(None)
        self.set_exec(exe)



def main():
    return

if __name__ == '__main__':
    main()
