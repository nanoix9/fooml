#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
import numpy as np
import fooml
from fooml.log import logger


def _merge(ga, phone):
    phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
    ga['brand'] = phone.phone_brand
    ga['model'] = phone.phone_brand.str.cat(phone.device_model)
    return ga

def _merge_app(events, appevents):
    device_apps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
            .groupby(['device_id','app_id'])['app_id'].agg(['size']))
    return device_apps

def _merge_label(device_apps, app_labels):
    device_labels = ((device_apps.reset_index())[['device_id','app_id']]
            .merge(app_labels[['app_id','label_id']])
            .groupby(['device_id','label_id'])['app_id'].agg(['size']))
    return device_labels

#def create_nn(X, y, *_, **__):
def create_nn(nb_feat):
    #logger.debug('create nnet with input dim %d' % X.shape[1])
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    #from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
    # create model
    model = Sequential()
    #model.add(Dense(10, input_dim=Xtrain.shape[1], init='normal', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(50, input_dim=X.shape[1], init='normal', activation='tanh'))
    model.add(Dense(50, input_dim=nb_feat, init='normal', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(20, init='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model


def main():
    use_dstv = False

    foo = fooml.FooML('talkingdata-mobile')

    foo.set_data_home('/vola1/scndof/data/talkingdata-mobile')
    foo.enable_data_cache()

    foo.load_csv('ga', train_path='gender_age_train.csv', test_path='gender_age_test.csv', target='group', index_col='device_id')
    foo.load_csv('phone', 'phone_brand_device_model.csv')
    foo.load_csv('events', 'events.csv', parse_dates=['timestamp'], index_col='event_id')
    foo.load_csv('appevents','app_events.csv', usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
    foo.load_csv('app_labels', 'app_labels.csv')

    drop_dup = fooml.feat_map('drop_dup', lambda df:df.drop_duplicates('device_id',keep='first').set_index('device_id'))
    feat_merge = fooml.feat_merge('merge', _merge)
    ga_dummy = fooml.trans('dummy_ga', 'dummy', opt=dict(cols=['brand', 'model'], sparse='csr'))

    dummy_event = fooml.new_comp('dummy_event', 'dummy', opt=dict(key='device_id', cols=['timestamp'], sparse='csr'))

    merge_app = fooml.feat_merge('merge_app', _merge_app)
    #app_align = fooml.trans('app_align', 'align_index')
    dummy_app = fooml.new_comp('dummy_app', 'dummy', opt=dict(key='device_id', cols=['app_id'], sparse='csr'))

    merge_label = fooml.feat_merge('merge_label', _merge_label)
    dummy_label = fooml.new_comp('dummy_label', 'dummy', opt=dict(key='device_id', cols=['label_id'], sparse='csr'))

    merge_all = fooml.new_comp('merge_all', 'merge')

    le = fooml.trans('targ_le', 'targetencoder')
    cate = fooml.trans('cate', 'to_categorical', args=[np.unique(foo.get_train_data('ga').y).shape[0]])

    lr = fooml.classifier('clf', 'LR', proba='only', opt=dict(C=0.02, multi_class='multinomial',solver='lbfgs'))
    xgbr = fooml.classifier('xgbr', 'xgboost', proba='only', opt=dict(params=dict()))

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import  fooml.comp.kr as kr
    callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                ]
    train_opt=dict(batch_size=16, nb_epoch=5, \
            verbose=1, shuffle=True, callbacks=callbacks)
    #nncls = fooml.nnet('nncls', model, train_opt=train_opt)
    #nncls = fooml.nnet('nncls', kr.Clf(fooml.LazyObj(create_nn, 'fit_generator'), train_opt=train_opt))
    nncls = fooml.nnet('nncls', create_nn(15853), train_opt=train_opt)
    use_dstv = True
    logloss = fooml.evaluator('logloss')

    #foo.add_comp(drop_dup, 'phone', 'phone_uniq')
    foo.add_comp(feat_merge, ['ga', 'phone'], 'ds_ga_merge')
    foo.add_comp(ga_dummy, 'ds_ga_merge', 'ds_ga_dummy')

    foo.add_comp(dummy_event, ['events', 'ds_ga_merge'], 'ds_event_dummy')

    foo.add_comp(merge_app, ['events', 'appevents'], 'ds_device_app')
    #foo.add_comp(app_align, ['merge_app', 'ds_merge'], 'app_align')
    foo.add_comp(dummy_app, ['ds_device_app', 'ds_ga_merge'], 'ds_app_dummy')

    foo.add_comp(merge_label, ['ds_device_app', 'app_labels'], 'ds_device_label')
    foo.add_comp(dummy_label, ['ds_device_label', 'ds_ga_merge'], 'ds_label_dummy')
    foo.add_comp(merge_all, ['ds_ga_dummy', 'ds_app_dummy', 'ds_label_dummy'], 'ds_all_dummy')
    foo.add_comp(le, 'ds_all_dummy', 'ds_targ_encoded')
    foo.add_comp(cate, 'ds_targ_encoded', 'ds_targ_cate')

    cv_clf = fooml.submodel('cv_clf', input='ds_targ_encoded', output=['y_proba', 'ds_logloss'])
    #cv_clf.add_comp(lr, 'ds_targ_encoded', 'y_proba')
    #cv_clf.add_comp(xgbr, 'ds_targ_encoded', 'y_proba')
    cv_clf.add_comp(nncls, 'ds_targ_encoded', 'y_proba')
    cv_clf.add_comp(logloss, 'y_proba', 'ds_logloss')

    cv = fooml.cross_validate('cv', cv_clf, eva='ds_logloss', k=5, use_dstv=use_dstv)
    #cv = fooml.cross_validate('cv', lr, k=2, evaluate=logloss)
    #foo.add_comp(cv, 'ds_targ_encoded', ['y_proba', 'ds_cv'])
    foo.add_comp(cv, 'ds_targ_cate', ['y_proba', 'ds_cv'])

    get_classes = lambda: foo.get_comp('targ_le')._obj.classes_
    foo.save_output('y_proba', opt=dict(label='device_id', columns=get_classes))

    #foo.desc_data()
    foo.compile()
    #foo.run_train()
    foo.run()

    return

if __name__ == '__main__':
    main()
