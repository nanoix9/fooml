#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

import fooml
from fooml import dataset

#img_size=(64, 64)
img_size = (224, 224)
color_type = 1

batch_size = 128
batch_size = 16
nb_epoch = 1
#nb_epoch = 5
#nb_epoch = 12
nb_epoch = 100
nb_epoch = 200
#nb_epoch = 500

def main():
    global img_size

    foo = fooml.FooML()
    foo.report_to('report.md')
    foo.enable_data_cache()

    data_suffix= '{}_{}_{}'.format(img_size[0], img_size[1], color_type)
    data_name = 'drive_' + data_suffix
    #data_name = 'drive_sample'
    #data_name = 'mnist'
    if data_name == 'drive_sample':
        global nb_epoch
        nb_epoch = 1
        foo.load_image_grouped(data_name, path='/vola1/scndof/data/drive/sample', resize=img_size)
    elif data_name.startswith('drive'):
        #foo.load_image_grouped(data_name, train_path='/vola1/scndof/data/drive/train', resize=img_size)
        foo.load_image_grouped(data_name, path='/vola1/scndof/data/drive', resize=img_size)
    #sys.exit()
    else:
        foo.use_data(data_name)

    ds_train = foo.get_train_data(data_name)
    #print ds_train.index
    #sys.exit()
    X_train, y_train = dataset.get_train(ds_train)
    img_size = X_train.shape[1:3]
    print img_size

    data_name_labeled = 'y_indexed'
    if len(y_train.shape) <= 1:
        foo.add_trans('le', 'labelencoder', input=data_name, output=data_name_labeled)
    else:
        data_name_labeled = data_name

    data_cate = 'y_cate'
    if len(y_train.shape) <= 1:
        foo.add_trans('cate', 'to_categorical', input=data_name_labeled , output=data_cate, args=[10])
    else:
        data_cate = data_name_labeled
    #data_cate = data_name_labeled

    data_reshape = 'x_reshape'
    if len(X_train.shape) < 4:
        foo.add_feat_trans('reshape',
                lambda data: data.reshape(data.shape[0], color_type, data.shape[1], data.shape[2]),
                input=data_cate, output=data_reshape)
    else:
        data_reshape = data_cate
    pred = 'y_pred'
    proba = 'y_proba' + data_suffix

    data_split = 'data_split'
    foo.add_trans('split', 'split', input=data_reshape, output=data_split, opt=dict(test_size=0.2))
    #data_split = data_reshape

    #foo.add_classifier('rand', 'random', input=data_reshape, output=[pred, proba], proba='with')
    model = create_model_v1(img_size, color_type)
    #model = create_model_v2(img_size, color_type)
    model = vgg_std16_model(img_size, color_type)

    foo.add_nn('nn', model, input=data_split, output=proba,
            train_opt=dict(batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True))
    #pred = data_name_labeled

    foo.evaluate('logloss', input=proba)

    #foo.add_inv_trans('invle', 'le', input=pred, output='y_pred_format')

    classes = lambda: foo.get_comp('le')._obj.classes_
    #foo.save_output(proba, path=sys.stdout, opt=dict(columns=classes))
    foo.save_output(proba, opt=dict(columns=classes))

    foo.show()
    foo.desc_data()
    foo.compile()

    #foo.run_train()
    foo.run()

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

    #model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def create_model_v2(img_size, color_type=1):
    img_rows, img_cols = img_size

    nb_classes = 10
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_std16_model(img_size, color_type=1):
    img_rows, img_cols = img_size

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    #model.load_weights('../input/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


if __name__ == '__main__':
    main()

