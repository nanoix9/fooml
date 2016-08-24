#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")

import fooml


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

def main():
    foo = fooml.FooML('talkingdata-mobile')

    foo.set_data_home('/vola1/scndof/data/talkingdata-mobile')
    foo.enable_data_cache()

    foo.load_csv('ga', 'gender_age_train.csv', target='group', index_col='device_id')
    foo.load_csv('phone', 'phone_brand_device_model.csv')
    foo.load_csv('events', 'events.csv', parse_dates=['timestamp'], index_col='event_id')
    foo.load_csv('appevents','app_events.csv', usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
    foo.load_csv('app_labels', 'app_labels.csv')

    drop_dup = fooml.feat_map('drop_dup', lambda df:df.drop_duplicates('device_id',keep='first').set_index('device_id'))
    feat_merge = fooml.feat_merge('merge', _merge)
    ga_dummy = fooml.trans('dummy_ga', 'dummy', opt=dict(cols=['brand', 'model'], sparse='csr'))
    merge_app = fooml.feat_merge('merge_app', _merge_app)
    #app_align = fooml.trans('app_align', 'align_index')
    dummy_app = fooml.new_comp('dummy_app', 'dummy', opt=dict(key='device_id', cols=['app_id'], sparse='csr'))
    merge_label = fooml.feat_merge('merge_label', _merge_label)
    dummy_label = fooml.new_comp('dummy_label', 'dummy', opt=dict(key='device_id', cols=['label_id'], sparse='csr'))
    merge_all = fooml.new_comp('merge_all', 'merge')
    le = fooml.trans('targ_le', 'targetencoder')
    lr = fooml.classifier('clf', 'LR', proba='only')
    logloss = fooml.evaluator('logloss')

    #foo.add_comp(drop_dup, 'phone', 'phone_uniq')
    foo.add_comp(feat_merge, ['ga', 'phone'], 'ds_ga_merge')
    foo.add_comp(ga_dummy, 'ds_ga_merge', 'ds_ga_dummy')

    foo.add_comp(merge_app, ['events', 'appevents'], 'ds_device_app')
    #foo.add_comp(app_align, ['merge_app', 'ds_merge'], 'app_align')
    foo.add_comp(dummy_app, ['ds_device_app', 'ds_ga_merge'], 'ds_app_dummy')

    foo.add_comp(merge_label, ['ds_device_app', 'app_labels'], 'ds_device_label')
    foo.add_comp(dummy_label, ['ds_device_label', 'ds_ga_merge'], 'ds_label_dummy')
    foo.add_comp(merge_all, ['ds_ga_dummy', 'ds_app_dummy', 'ds_label_dummy'], 'ds_all_dummy')
    foo.add_comp(le, 'ds_all_dummy', 'ds_targ_encoded')
    foo.add_comp(lr, 'ds_targ_encoded', 'y_proba')
    foo.add_comp(logloss, 'y_proba')

    #foo.desc_data()
    foo.compile()
    foo.run_train()

    return

if __name__ == '__main__':
    main()
