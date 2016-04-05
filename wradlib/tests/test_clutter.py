#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.clutter as cl
import unittest


# -------------------------------------------------------------------------------
# testing the filter helper function
# -------------------------------------------------------------------------------
class TestClutter(unittest.TestCase):
    # def test_filter_gabella_a_trdefault(self):
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data)
    #     self.assertTrue(result == 4)
    #
    # def test_filter_gabella_a_tr1(self):
    #     pass
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data, tr1=5)
    #     self.assertTrue(result == 3)
    # -------------------------------------------------------------------------------
    # testing the first part of the filter
    # -------------------------------------------------------------------------------
    def filter_setup(self):
        img = np.zeros((36, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[9:12, 4:7] = 11  # precip field
        self.img = img
        pass

    def test_filter_gabella_a(self):
        pass

    def test_filter_window_distance(self):
        self.filter_setup()
        self.img[15:17, 5:7] = np.nan  # nans
        clutter = self.img.copy()
        clutter[self.img > 0] = True
        clutter[self.img == 11] = False
        clutter[np.isnan(self.img)] = False
        np.set_printoptions(precision=2)
        rscale = 250
        similar = cl.filter_window_distance(self.img, rscale, fsize=300, tr1=4)
        result = similar < 0.3
        np.set_printoptions(precision=3)
        self.assertTrue((result == clutter).all())
