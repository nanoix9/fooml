#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


class Component(object):

    def __init__(self, name, obj):
        self.name = name
        self.obj = obj

class _CompList(Component):

    def __init__(self, name):
        super(_CompList, self).__init__(name, [])

    def add_obj(self, name, obj):
        self.obj.append(Component(name, obj))

    def add_component(self, name, comp):
        self.obj.append(comp)

    def __iter__(self):
        for o in self.obj:
            yield o

class Parallel(_CompList):

    def __init__(self, name='parallel'):
        super(Parallel, self).__init__(name)
        #self._objs = (name, [])

class Serial(_CompList):

    def __init__(self, name='serial'):
        super(Serial, self).__init__(name)
        #self._objs = []

def main():
    return

if __name__ == '__main__':
    main()
