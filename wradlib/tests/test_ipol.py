# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_ipol
# Purpose:     unit tests for the wrdalib.ipol module
#
# Author:      Thomas Pfaff
#
# Created:     31.05.2012
# Copyright:   (c) Thomas Pfaff 2012
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import numpy as np

import wradlib


def test_OrdinaryKriging_1():
    """testing the basic behaviour of the OrdinaryKriging class"""
    src = np.array([[0.,0.], [4.,0]])
    trg = np.array([[0.,0.], [2.,0.], [1.,0], [4.,0]])

    ip = wradlib.ipol.OrdinaryKriging(src, trg, '1.0 Lin(2.0)')

    vals = np.array([[1., 2., 3.],
                     [3., 2., 1.]])
    res = ip(vals)
    assert np.all(res == np.array([[ 1.,   2.,   3. ],
                                   [ 2.,   2.,   2. ],
                                   [ 1.5,  2.,   2.5],
                                   [ 3.,   2.,   1. ]]))


if __name__ == '__main__':
    pass
    #test_OrdinaryKriging_1()
