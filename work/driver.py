#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./")

import fooml


def main():
    foo = fooml.FooML()

    data_name = 'drive'
    foo.load_image_grouped(data_name, train_path='/vola1/scndof/data/drive/sample', resize=(64, 64))
    #foo.load_image_grouped('driver', path='~/data/driver')

    data_name_labeled = 'y_indexed'
    foo.add_trans('le', 'labelencoder', input=data_name, output=data_name_labeled)
    foo.add_inv_trans('invle', 'le', input=data_name_labeled, output='_')

    foo.show()
    foo.desc_data()
    foo.compile()

    return

if __name__ == '__main__':
    main()
