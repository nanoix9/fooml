#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.sparse as sp

import comp
import mixin
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger

class KerasComp(comp.Comp):
    pass

class Clf(KerasComp):

    def __init__(self, obj, train_opt={}):
        super(KerasComp, self).__init__(obj)
        self._train_opt = train_opt
        self.__batch_size = 64

    def fit(self, data):
        X_train, y_train, X_valid, y_valid = dataset.get_train_valid(data)
        opt = dict(self._train_opt)
        if X_valid is not None:
            logger.info('trainning nerual network with validation set: X%s, y%s' % (X_valid.shape, y_valid.shape))
        else:
            logger.info('trainning nerual network with no validation set')
        if isinstance(X_train, (sp.spmatrix)):
            return self._fit_sparse(X_train, y_train, X_valid, y_valid, opt)
        else:
            return self._fit_dense(X_train, y_train, X_valid, y_valid, opt)

    def _fit_dense(self, X_train, y_train, X_valid, y_valid, opt):
        logger.info('and options: %s' % str(opt))
        if X_valid is not None:
            opt['validation_data'] = (X_valid, y_valid)
        return self.get_obj().fit(X_train, y_train, **opt)

    def _fit_sparse(self, X_train, y_train, X_valid, y_valid, opt):
        batch_size = opt.pop('batch_size', 64)
        shuffle = opt.pop('shuffle', True)
        opt['samples_per_epoch'] = X_train.shape[0]
        logger.info('fit_generator with sparse matrix: batch_size %d, shuffle %s' \
                'and options: %s' % (batch_size, shuffle, str(opt)))
        if X_valid is not None:
            opt['validation_data'] = (X_valid.toarray(), y_valid)
        fit = self.get_obj().fit_generator(generator=Clf._batch_generator(X_train, y_train, batch_size, shuffle), **opt)
        self.__batch_size = batch_size
        return fit

    def trans(self, data):
        X, y = data
        if isinstance(X, (sp.spmatrix)):
            batch_size = self.__batch_size
            val_samples = X.shape[0]
            logger.info('predict_generator with sparse matrix: batch_size %d ' \
                    'and val_samples: %d' % (batch_size, val_samples))
            score = self.get_obj().predict_generator(generator=Clf._batch_generator(X, batch_size=batch_size, shuffle=False), val_samples=val_samples)
        else:
            score = self.get_obj().predict(X)
        return dataset.dssy(score, y, data.index)

    def fit_trans(self, data):
        if isinstance(data, dataset.dstv):
            ds_train, ds_valid = data
        else:
            ds_train = data
        self.fit(data)
        return self.trans(ds_train)

        #print X, y
        #self.get_obj().fit_transform(X, y)
        #print self.get_obj()
        #return self.get_obj().fit_transform(X, y)

    @staticmethod
    def _batch_generator(X, y=None, batch_size=16, shuffle=True):
        #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
        number_of_batches = np.ceil(float(X.shape[0])/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        #print number_of_batches
        #print sample_index
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = X[batch_index,:].toarray()
            #print counter, batch_size*counter, batch_size*(counter+1), X_batch.shape
            if y is not None:
                y_batch = y[batch_index]
                #print X_batch, y_batch
                yield X_batch, y_batch
            else:
                yield X_batch
            counter += 1
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

class Eva(mixin.EvaMixin, KerasComp):

    def __init__(self, obj):
        func, args, opt = obj
        super(Eva, self).__init__(func)
        self.args = args
        self.opt = opt

    def _cal_func(self, y, score):
        return self.get_obj()(y, score, *self.args, **self.opt)


def main():
    return

if __name__ == '__main__':
    main()
