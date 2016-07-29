#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from fooml import dataset
from fooml.dt import slist
from fooml import util
from fooml.log import logger


class BaseMixin(object):

    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)

class TransMixin(BaseMixin):

    def fit(self, data):
        return self._apply(data, self._fit_func)

    def fit_trans(self, data):
        if isinstance(data, dataset.dstv):
            return dataset.dstv(self._apply(data.train, self._fit_trans_func),
                    self._apply(data.valid, self._trans_func))
        else:
            return self._apply(data, self._fit_trans_func)

    def trans(self, data):
        return self._apply(data, self._trans_func)

class TargTransMixin(TransMixin):

    def _apply(self, data, func):
        return dataset.mapy(func, data)

class FeatTransMixin(TransMixin):

    def _apply(self, data, func):
        return dataset.mapx(func, data)

class SplitMixin(BaseMixin):

    def fit_trans(self, data):
        if isinstance(data, dataset.dstv):
            return data
        elif not isinstance(data, dataset.dsxy):
            raise TypeError('data is not dsxy type')
        X, y = data
        Xt, Xv, yt, yv, it, iv = self._split(X, y, data.index)
        return dataset.dstv(dataset.dsxy(Xt, yt, it), dataset.dsxy(Xv, yv, iv))

    def trans(self, data):
        return data

class PartSplitMixin(SplitMixin):

    def fit_trans(self, data):
        return self._split(data)

    def _split(self, data):
        main_data, labels, label_index = self.__get_data_labels(data)
        if isinstance(main_data, dataset.dstv):
            return main_data
        it, iv = self.__get_split_index(main_data, labels, label_index, by=None)
        return dataset.split(main_data, it, iv)

    def _iter_split(self, data):
        main_data, labels, label_index = self.__get_data_labels(data)
        for it, iv in self.__iter_split_index(main_data, labels, label_index, by=None):
            yield dataset.split(main_data, it, iv)

    def __get_data_labels(self, data):
        main_data, partition = self._get_main_and_labels(data)

        if isinstance(main_data, dataset.dstv):
            return main_data, None, None
        elif not isinstance(main_data, dataset.dsxy):
            raise TypeError('data is not dsxy type')
        if not isinstance(partition, dataset.dsxy):
            raise TypeError('partition is not dsxy type')
        labels, label_index = self._get_labels(main_data, partition)
        return main_data, labels, label_index

    def _get_main_and_labels(self, data):
        if isinstance(data, list):
            main_data = data[0]
            partition = data[1]
        elif isinstance(data, dataset.dataset):
            main_data = data
            partition = data
        else:
            raise TypeError()
        return main_data, partition

    def __get_split_index(self, main_data, labels, label_index, by):
        self.__validate_data(main_data, labels, label_index)
        it_label, iv_label = self._split_labels_index(labels)
        return self.__align_index(it_label, main_data.index, label_index), \
                self.__align_index(iv_label, main_data.index, label_index)

    def __iter_split_index(self, main_data, labels, label_index, by):
        self.__validate_data(main_data, labels, label_index)
        for it_label, iv_label in self._split_labels_index(labels):
            yield self.__align_index(it_label, main_data.index, label_index), \
                   self.__align_index(iv_label, main_data.index, label_index)

    def __align_index(self, idx_label, data_index, label_index):
        if data_index is None or label_index is None:
            return idx_label
        #print tk, vk
        label_index_set = set(label_index[idx_label])
        idx_data = np.tile(False, data_index.shape)
        for i, idx in enumerate(data_index):
            if idx in label_index_set:
                idx_data[i] = True
        return idx_data

    def _value_to_index(self, value_select, value_all):
        vk_set = set(value_select)
        #print vk_set
        idx = []
        for i, v in enumerate(value_all):
            if value_all[i] in vk_set:
                idx.append(i)
        return np.array(idx)

    def __xx_align_index(self, idx_label, idx_data, label_index):
        index = main_data.index
        if index is None or label_index is None:
            return tk, vk
        #print tk, vk
        v_idx_set = set()
        vk_set = set(vk)
        #print labels
        #print vk_set
        for i, idx in enumerate(label_index):
            if labels[i] in vk_set:
                v_idx_set.add(idx)
        #print index
        #print index.shape
        iv = np.tile(False, index.shape)
        #print iv
        #print iv.shape
        for i, idx in enumerate(index):
            if idx in v_idx_set:
                iv[i] = True
        it = np.logical_not(iv)
        #print iv
        #print it
        #print yv, vidx
        #print yt, tidx
        return it, iv

    def __validate_data(self, main_data, labels, label_index):
        if label_index is not None and labels.shape[0] != label_index.shape[0]:
            raise RuntimeError('key and index are not same size')

    def trans(self, data):
        return data[0]

#class ClassifierMixin(BaseMixin):
#
#    #def fit(self, data):
#
#    def trans(self, ds):

class EvaMixin(BaseMixin):

    def fit(self, data):
        pass

    def trans(self, data):
        eva_list = []
        for d in slist.iter_multi(data, strict=True):
            if not isinstance(d, (dataset.dssy, dataset.dscy)):
                raise TypeError('data must be dssy or dscy type')
            score, y = d
            if y is None:
                eva = np.nan
            else:
                eva = self._cal_func(y, score)
            eva_list.append(str(eva))
        eva_str = util.joins(eva_list)
        if '\n' not in eva_str:
            return dataset.desc('scores: ' + eva_str)
        else:
            return dataset.desc('scores:\n' + util.indent(eva_str))


def main():
    return

if __name__ == '__main__':
    main()
