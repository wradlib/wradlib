#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:        test_dp.py
# Purpose:     testing file for the wradlib.dp module
#
# Authors:     wradlib developers
#
# Created:     26.02.2016
# Copyright:   (c) wradlib developers
# Licence:     The MIT License
# -------------------------------------------------------------------------------

import unittest

import wradlib.dp as dp
import numpy as np

class KDPFromPHIDPTest(unittest.TestCase):
    def setUp(self):
        self.kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
        phidp_true = np.cumsum(self.kdp_true)
        self.phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
        gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
        self.phidp_raw[gaps] = np.nan

    def test_process_raw_phidp_vulpiani(self):
        pass

    def test_unfold_phi_vulpiani(self):
        pass

    def test_kdp_from_phidp_finitediff(self):
        kdp_re = dp.kdp_from_phidp_finitediff(self.phidp_raw)
        pass

    def test_kdp_from_phidp_linregress(self):
        kdp_re = dp.kdp_from_phidp_linregress(self.phidp_raw)
        pass

    def test_kdp_from_phidp_sobel(self):
        kdp_re = dp.kdp_from_phidp_sobel(self.phidp_raw)
        pass

    def test_kdp_from_phidp_convolution(self):
        kdp_re = dp.kdp_from_phidp_convolution(self.phidp_raw)
        pass

class TextureTest(unittest.TestCase):
    def test_texture(self):
        pass

if __name__ == '__main__':
    unittest.main()
