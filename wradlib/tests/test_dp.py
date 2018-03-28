#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.dp as dp
import numpy as np


class KDPFromPHIDPTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic truth
        self.dr = 0.5
        r = np.arange(0, 100, self.dr)
        self.kdp_true = np.sin(0.3 * r)
        self.kdp_true[self.kdp_true < 0] = 0.
        self.phidp_true = np.cumsum(self.kdp_true) * 2 * self.dr
        # Synthetic observation of PhiDP with a random noise and gaps
        self.phidp_raw = (self.phidp_true +
                          np.random.uniform(-2, 2, len(self.phidp_true)))
        self.gaps = np.random.uniform(0, len(r), 20).astype("int")
        self.phidp_raw[self.gaps] = np.nan
        self.rho = np.random.uniform(0.8, 1.0, len(r))

    def test_process_raw_phidp_vulpiani(self):
        dp.process_raw_phidp_vulpiani(self.phidp_raw, dr=self.dr,
                                      copy=True)
        dp.process_raw_phidp_vulpiani(self.phidp_raw, dr=self.dr)

    def test_kdp_from_phidp(self):
        dp.kdp_from_phidp(self.phidp_raw, dr=self.dr)
        dp.kdp_from_phidp(self.phidp_raw, dr=self.dr, method='slow')

    def test_linear_despeckle(self):
        dp.linear_despeckle(self.phidp_raw, ndespeckle=3, copy=True)
        dp.linear_despeckle(self.phidp_raw, ndespeckle=5, copy=True)

    def test_unfold_phi_naive(self):
        dp.unfold_phi_naive(self.phidp_raw, self.rho)
        dp.unfold_phi_naive(self.phidp_raw, self.rho, copy=True)

    def test_unfold_phi_vulpiani(self):
        dp.unfold_phi_vulpiani(self.phidp_raw, self.kdp_true)

    def test__fill_sweep(self):
        dp._fill_sweep(self.phidp_raw, kind='linear')


class TextureTest(unittest.TestCase):
    def setUp(self):
        img = np.zeros((360, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[60:120, 2:7] = 11  # precip field
        self.r = np.arange(0, 100000, 10000)
        self.az = np.arange(0, 360)
        self.img = img

    def test_texture(self):
        dp.texture(self.img)


if __name__ == '__main__':
    unittest.main()
