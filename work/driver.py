#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint

#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
#from keras.models import model_from_json

import fooml
from fooml import dataset
from fooml import util

do_cv = True
#img_size=(64, 64)
img_size = (224, 224)
color_type = 1
color_type = 3

batch_size = 128
batch_size = 16
nb_epoch = 1
nb_epoch = 5
#nb_epoch = 12
#nb_epoch = 15
#nb_epoch = 100
#nb_epoch = 200
#nb_epoch = 500

def main(test):
    global img_size
    global color_type

    foo = fooml.FooML()
    foo.report_to('report.md')

    data_suffix= '_{}x{}x{}'.format(img_size[0], img_size[1], color_type)
    data_name = 'drive' + data_suffix
    if test:
        data_name = 'drive_sample' + data_suffix
    #data_name = 'mnist'
    if data_name.startswith('drive_sample'):
        global nb_epoch
        nb_epoch = 1
        foo.load_image(data_name, path='/vola1/scndof/data/drive/sample', train_type='grouped', resize=img_size, color_type=color_type)
    elif data_name.startswith('drive'):
        foo.enable_data_cache()
        #foo.load_image_grouped(data_name, train_path='/vola1/scndof/data/drive/train', resize=img_size)
        foo.load_image(data_name, path='/vola1/scndof/data/drive', train_type='grouped', resize=img_size, color_type=color_type)
    #sys.exit()
    elif data_name == 'mnist':
        img_size = (224, 224)
        color_type = 1
        foo.use_data(data_name)

    data_driver_id = 'driver_id'
    foo.load_csv(data_driver_id, train_path='/vola1/scndof/data/drive/driver_imgs_list.csv', index_col=2)

    ds_train = foo.get_train_data(data_name)
    #print ds_train.index
    #sys.exit()
    X_train, y_train = dataset.get_train(ds_train)
    img_size = X_train.shape[1:3]
    print img_size

    if len(y_train.shape) <= 1 and data_name != 'mnist':
        le = fooml.trans('le', 'targetencoder')
    else:
        le = fooml.nop()

    if len(y_train.shape) <= 1:
        cate = fooml.trans('cate', 'to_categorical', args=[10])
    else:
        cate = fooml.nop()

    if len(X_train.shape) < 4 or color_type == 1:
        reshape = fooml.feat_trans('reshape1',
                lambda data: data.reshape(data.shape[0], color_type, data.shape[1], data.shape[2]))
    elif color_type == 3:
        reshape = fooml.trans('reshape3', 'vgg_preproc')
    else:
        reshape = fooml.nop()

    data_name_labeled = 'y_indexed'
    data_cate = 'y_cate'
    data_reshape = 'x_reshape'
    pred = 'y_pred'
    proba = 'y_proba' + data_suffix

    foo.add_comp(le, input=data_name, output=data_name_labeled)
    foo.add_comp(cate, input=data_name_labeled , output=data_cate)
    foo.add_comp(reshape, input=data_cate, output=data_reshape)

    #model = create_model_v1(img_size, color_type)
    #model = create_model_v2(img_size, color_type)
    #model = vgg_std16_model(img_size, color_type)
    #model = vgg_19()
    #model.summary()
    callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                ]
    train_opt=dict(batch_size=batch_size, nb_epoch=nb_epoch, \
            verbose=1, shuffle=True, callbacks=callbacks)
    #nncls = fooml.nnet('nncls', model, train_opt=train_opt)
    nncls = fooml.nnet('nncls', 'vgg19', \
            opt=dict(nb_class=10, weight_path='/vola1/scndof/model/vgg19_weights.h5'), \
            train_opt=train_opt)
    logloss = fooml.evaluator('logloss')
    if not do_cv:
        data_split = 'data_split'
        #foo.add_trans('split', 'split', input=data_reshape, output=data_split, opt=dict(test_size=0.2))
        split = fooml.splitter('split', partition='driver_id', part_key=lambda df: df.subject, opt=dict(test_size=0.2))
        foo.add_comp(split, input=[data_reshape, 'driver_id'], output=data_split)
        #data_split = data_reshape

        #foo.add_classifier('rand', 'random', input=data_reshape, output=[pred, proba], proba='with')
        foo.add_comp(nncls, input=data_split, output=proba)
        #pred = data_name_labeled

        foo.add_comp(logloss, input=proba)

        #foo.add_inv_trans('invle', 'le', input=pred, output='y_pred_format')

        classes = lambda: foo.get_comp('le')._obj.classes_
        #foo.save_output(proba, path=sys.stdout, opt=dict(columns=classes))
        foo.save_output(proba, opt=dict(label='img', columns=classes))
    else:
        sub = fooml.submodel('submdl', input=data_reshape, output='logloss')
        sub.add_comp(nncls, input=data_reshape, output=proba)
        sub.add_comp(logloss, input=proba, output='logloss')
        cv = fooml.cross_validate('cv', sub, k=2, type='labelkfold', \
                label='driver_id', label_key=lambda df: df.subject, \
                use_dstv=True)

        foo.add_comp(cv, input=[data_reshape, 'driver_id'])

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

if __name__ == '__main__':
    test = False
    #print sys.argv
    if len(sys.argv) >= 2 and sys.argv[1] == 'test':
        test = True
    #print test
    main(test)

