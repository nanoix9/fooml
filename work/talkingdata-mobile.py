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
    deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
            .groupby(['device_id','app_id'])['app_id'].agg(['size']))
    return deviceapps

def main():
    foo = fooml.FooML('talkingdata-mobile')

    foo.set_data_home('/vola1/scndof/data/talkingdata-mobile')
    foo.enable_data_cache()

    foo.load_csv('ga', 'gender_age_train.csv', index_col='device_id')
    foo.load_csv('phone', 'phone_brand_device_model.csv')
    foo.load_csv('events', 'events.csv', parse_dates=['timestamp'], index_col='event_id')
    foo.load_csv('appevents','app_events.csv', usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})

    drop_dup = fooml.feat_trans('drop_dup', lambda df:df.drop_duplicates('device_id',keep='first').set_index('device_id'))

    feat_merge = fooml.feat_merge('merge', _merge)
    ga_dummy = fooml.trans('ga_dummy', 'dummy', opt=dict(cols=['brand', 'model'], sparse='csr'))
    app_merge = fooml.feat_merge('app_merge', _merge_app)
    app_align = fooml.trans('app_align', 'align_index')
    app_dummy = fooml.trans('app_dummy', 'dummy', opt=dict(key='device_id', cols=['app_id'], sparse='csr'))

    #foo.add_comp(drop_dup, 'phone', 'phone_uniq')
    foo.add_comp(feat_merge, ['ga', 'phone'], 'ds_merge')
    foo.add_comp(ga_dummy, 'ds_merge', 'ds_dummy')
    foo.add_comp(app_merge, ['events', 'appevents'], 'ds_app_merge')
    #foo.add_comp(app_align, ['app_merge', 'ds_merge'], 'app_align')
    foo.add_comp(app_dummy, ['ds_app_merge', 'ds_merge'], 'ds_app_dummy')

    #foo.desc_data()
    foo.compile()
    foo.run_train()

    return

if __name__ == '__main__':
    main()
