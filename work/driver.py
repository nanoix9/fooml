#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json

import fooml
img_size=(64, 64)
color_type = 1

batch_size = 256
nb_epoch = 1
nb_epoch = 12

def main():
    foo = fooml.FooML()

    data_name = 'drive'
    #foo.load_image_grouped(data_name, train_path='/vola1/scndof/data/drive/sample', resize=(64, 64))
    foo.load_image_grouped(data_name, train_path='/vola1/scndof/data/drive/train', resize=img_size)
    #foo.load_image_grouped('driver', path='~/data/driver')

    data_name_labeled = 'y_indexed'
    foo.add_trans('le', 'labelencoder', input=data_name, output=data_name_labeled)
    data_cate = 'y_cate'
    data_reshape = 'x_reshape'
    foo.add_trans('cate', 'to_categorical', input=data_name_labeled , output=data_cate, args=[10])
    foo.add_feat_trans('reshape',
            lambda data: data.reshape(data.shape[0], color_type, data.shape[1], data.shape[2]),
            input=data_cate , output=data_reshape)
    pred = 'y_pred'
    proba = 'y_proba'

    #foo.add_classifier('rand', 'random', input=data_name_labeled, output=[pred, proba], proba='with')
    #pred = data_name_labeled
    model = create_model_v1(img_size, color_type)
    foo.add_nn('nn', model, input=data_reshape, output=proba,
            train_opt=dict(batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
            )

    foo.evaluate('logloss', input=proba)

    #foo.add_inv_trans('invle', 'le', input=pred, output='y_pred_format')

    foo.show()
    foo.desc_data()
    foo.compile()

    foo.run_train()

    return


def create_model_v1(img_size, color_type=1):
    img_rows, img_cols = img_size

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model


if __name__ == '__main__':
    main()
