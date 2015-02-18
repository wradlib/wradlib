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
    @util.apichange("0.6.0", par='x', msg="x will take int data in the future")
    def test_function1(z, x=None):
        print(x, type(x))
        sys.stdout.flush()
        return z, x

    @util.apichange("0.6.0", par='x', typ=str, exfunc=help_function)
    def test_function2(z, x=None):
        print(x, type(x))
        sys.stdout.flush()
        return z, x

    @util.apichange("0.6.0", par='x', expar='y', typ=str, exfunc=help_function)
    def test_function3(z, y=None):
        print(y, type(y))
        sys.stdout.flush()
        return z, y

    # this will run normal, because x is no parameter
    test_function1(1)
    sys.stderr.flush()
    time.sleep(1)

    # this will display the warning message, because x is parameter
    test_function1(1, x='10')
    sys.stderr.flush()
    time.sleep(1)

    # this will display the warning message, because x is parameter
    test_function1(1, x=20)
    sys.stderr.flush()
    time.sleep(1)

    # this will display the warning message, because x is string,
    # use the help_function to convert to int type
    test_function2(1, x='30')
    sys.stderr.flush()
    time.sleep(1)

    # this will run normal, because x is no string parameter
    test_function2(1, x=40)
    sys.stderr.flush()
    time.sleep(1)


    # this will display the warning message, because x is string
    # use the help_function to convert to int type
    # switch par (x) to expar (y)
    test_function3(1, x='50')
    sys.stderr.flush()
    time.sleep(1)

    # this will display the warning message, because x is parameter
    # and expar is given
    # switch par (x) to expar (y)
    test_function3(1, x=60)
    sys.stderr.flush()

if __name__ == '__main__':
    ex_apichange()