# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------------------
# Name:        apichange_example
# Purpose:
#
# Author:      Kai Muehlbauer
#
# Created:     18.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import time, sys
import wradlib.util as util


def help_function(x):
    return int(x)


def ex_apichange():
    @util.apichange_kwarg("0.6.0", par='x', typ=str, msg="x will take int data in the future")
    def futurechange(z, x=None):
        if isinstance(x, str):
            print(z, x, type(x), "normal function behaviour, DeprecationWarning is issued")
        elif isinstance(x, type(None)):
            print(z, x, type(x), "normal function behaviour, no DeprecationWarning")
        else:
            print(z, x, type(x), "using wrong type here, no DeprecationWarning, "
                              "but TypeError will be raised")
            raise TypeError("Wrong Input %s, 'str' expected" % type(x))
        sys.stdout.flush()
        return z, x

    futurechange(0)
    time.sleep(1)
    futurechange(1, x='10')
    time.sleep(1)
    try:
        futurechange(2, x=20)
        time.sleep(1)
    except TypeError as e:
        print "Type error: {0}".format(e)


    @util.apichange_kwarg("0.6.0", par='x', typ=str, exfunc=help_function)
    def typechanged(z, x=None):
        if isinstance(x, int):
            print(z, x, type(x), "normal function behaviour or type change, "
                              "DeprecationWarning is issued when 'x' is type(str)")
        elif isinstance(x, type(None)):
            print(z, x, type(x), "normal function behaviour, no DeprecationWarning")
        else:
            print(z, x, type(x), "using wrong type here, TypeError will be raised")
            raise TypeError("Wrong Input %s, 'int' expected" % type(x))
        sys.stdout.flush()
        return z, x

    print("Test typechange_kwarg")
    typechanged(0)
    time.sleep(1)
    typechanged(3, x='30')
    time.sleep(1)
    typechanged(4, x=40)
    time.sleep(1)


    @util.apichange_kwarg("0.6.0", par='x', typ=str, expar='y')
    def namechanged(z, y=None):
        if isinstance(y, str):
            print(z, y, type(y), "DeprecationWarning")
        elif isinstance(y, type(None)):
            print(z, y, type(y), "normal function behaviour, no DeprecationWarning")
        else:
            print(z, y, type(y), "using wrong type here, TypeError is issued")
            raise TypeError("Wrong Input %s, 'str' expected" % type(y))
        sys.stdout.flush()
        return z, y

    print("Test namechange_kwarg")
    namechanged(0)
    time.sleep(1)
    namechanged(5, x='50')
    time.sleep(1)
    try:
        namechanged(6, x=60)
    except TypeError as e:
        print "Type error: {0}".format(e)
    time.sleep(1)
    namechanged(7, y='70')
    time.sleep(1)
    try:
        namechanged(8, y=80)
    except TypeError as e:
        print "Type error: {0}".format(e)
    time.sleep(1)

    @util.apichange_kwarg("0.6.0", par='x', typ=str, expar='y', exfunc=help_function)
    def name_and_type_changed(z, y=None):
        if isinstance(y, int):
            print(z, y, type(y), "normal function behaviour or paramter and type change, "
                              "DeprecationWarning is issued when 'x' is given")
        elif isinstance(y, type(None)):
            print(z, y, type(y), "normal function behaviour, no DeprecationWarning")
        else:
            print(z, y, type(y), "using wrong type here, TypeError will be raised")
            raise TypeError("Wrong Input %s, 'str' expected" % type(y))
        return z, y

    print("Test apichange_kwarg type and name")
    name_and_type_changed(0)
    time.sleep(1)
    name_and_type_changed(9, x='90')
    time.sleep(1)
    try:
        name_and_type_changed(10, x=100)
    except TypeError as e:
        print "Type error: {0}".format(e)
    time.sleep(1)


if __name__ == '__main__':
    ex_apichange()