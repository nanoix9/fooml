#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import comp
import mixin
import group
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger
from sklearn.preprocessing import LabelEncoder

class SkComp(comp.Comp):

    def __init__(self, obj):
        super(SkComp, self).__init__(obj)

    def trans(self, data):
        return self._obj.transform(data)

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

        #print X, y
        #self._obj.fit_transform(X, y)
        #print self._obj
        #return self._obj.fit_transform(X, y)

class TargMap(mixin.TargMapMixin, SkComp):

    def __init__(self, obj):
        super(TargMap, self).__init__(obj)
        self._fit_func = self._obj.fit
        self._trans_func = self._obj.transform

    def _fit_trans_func(self, *args, **kwds):
        ret = self._obj.fit_transform(*args, **kwds)
        if isinstance(self._obj, LabelEncoder):
            logger.info('labels: {}'.format(self._obj.classes_))
        return ret

class TargInvTrans(mixin.TargMapMixin, SkComp):

    def __init__(self, another):
        super(TargInvTrans, self).__init__(another._obj)

    #def fit(self, y):
    #    raise RuntimeError('TargInvTrans cannot be fitted')

    def _fit_trans_func(self, y):
        return self._trans_func(y)

    def _trans_func(self, y):
        return self._obj.inverse_transform(y)

class Clf(SkComp):

    def __init__(self, obj, proba=None):
        super(Clf, self).__init__(obj)
        self._cal_proba = proba == 'with' or proba == 'only'
        self._cal_class = proba != 'only'

    def fit(self, data):
        X, y = data
        return self._obj.fit(X, y)

    def trans(self, ds):
        assert(isinstance(ds, dataset.dsxy))
        X, y = ds
        sy = cy = None
        if self._cal_proba:
            score = self._predict_proba(X)
            sy = dataset.dssy(score, y, ds.index)
        if self._cal_class:
            cls = self._obj.predict(X)
            cy = dataset.dscy(cls, y, ds.index)

        if cy is not None and sy is not None:
            return [cy, sy]
        elif sy is not None:
            return sy
        else:
            return cy

    def _predict_proba(self, X):
        if hasattr(self._obj, 'decision_function'):
            score = self._obj.decision_function(X)
        else:
            logger.info('no "decision_function" found, use "predict_proba" instead')
            score = self._obj.predict_proba(X)
            # if it is a binary classification problem, return a 1-D array
            if score.shape[1] == 2:
                score = score[:,1]
        #print '>>>>>', score
        #print '>>>>>', self._obj.classes_
        #sys.exit()
        # of probablities of class 1
        #    #print score
        #sys.exit()
        return score

class Dummy(Clf):

    def trans(self, ds):
        X, y = ds
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return Clf.trans(self, dataset.dsxy(X, y, ds.index))

class Eva(mixin.EvaMixin, SkComp):

    def __init__(self, obj):
        func, args, opt = obj
        super(Eva, self).__init__(func)
        self.args = args
        self.opt = opt

    def _cal_func(self, y, score):
        return self._obj(y, score, *self.args, **self.opt)

class CV(group.ExecMixin, mixin.PartSplitMixin, SkComp):

    def __init__(self, obj, k=5, exe=None, \
            label=None, label_key=lambda x:x,
            use_dstv=False):
        func, args, opt = obj
        super(CV, self).__init__(func)
        self.type = func.__name__
        self.k = k
        self._label = label
        self._label_key = label_key
        self._use_dstv = use_dstv
        if exe is not None:
            self.set_exec(exe)

    def fit_trans(self, data):
        if not isinstance(data, dataset.dsxy) \
                and not all(isinstance(d, dataset.dsxy) for d in data):
            raise TypeError('data must be dsxy or list of dsxy, got: %s' % util.get_type_fullname(data))
        out = []
        out_names = self._exec._cgraph._graph._out
        for i, ds in enumerate(self._iter_split(data)):
            logger.info('cross validation round %d out of %d: %d(train)/%d(test) samples' \
                    % (i+1, self.k, ds.train.nsamples(), ds.valid.nsamples()))
            if self._use_dstv:
                ret_train = self._exec.run_train(ds)
                ret_test = None
            else:
                ret_train = self._exec.run_train(ds.train)
                ret_test = self._exec.run_test(ds.valid)

            res_list = []
            res_list.append('training result:')
            def _format(res):
                return ['%s:\n%s' % (slist.get(out_names, i), util.indent(str(r), 2)) \
                        for i, r in slist.enumerate(res)]
            res_list.append(_format(ret_train))
            if ret_test is not None:
                res_list.append('testing result:')
                res_list.append(_format(ret_test))
            outi = []
            outi.append('cross validation %d' % (i+1))
            outi.append(res_list)
            out.append(outi)
        return dataset.desc(out)

    def trans(self, data):
        main_data, _ = self._get_main_and_labels(data)
        return self._exec.run_test(main_data)

    def _get_labels(self, data, label_data):
        if self._label is None:
            return data.y, data.get_index()
        else:
            return np.array(self._label_key(label_data.X)), label_data.get_index()

    def _split_labels_index(self, labels):
        return self._get_iter(labels)

    def _get_iter(self, data):
        #print data
        if self.type == 'KFold':
            return self._obj(len(data), self.k)
        elif self.type == 'StratifiedKFold':
            return self._obj(data, self.k)
        elif self.type == 'LabelKFold':
            return self._obj(data, self.k)
        else:
            raise RuntimeError()

    def _extr_desc(self):
        return 'num folds: %d\nsubmodel:\n%s' % \
                (self.k, util.indent(str(self._exec), 2))


def _test_split():
    import sklearn.cross_validation as cv
    c = CV((cv.KFold, [], {}), exe=lambda:0)
    d = dataset.dsxy(np.arange(10, 15), np.arange(1, 6))
    c._exec._cgraph = lambda:0
    c._exec._cgraph._graph = lambda:0
    c._exec._cgraph._graph._out = 'out'
    c._exec.run_train = lambda *x: 1001
    c._exec.run_test = lambda *x: 1002
    print c.fit_trans(d)

def main():
    _test_split()
    return

if __name__ == '__main__':
    main()
