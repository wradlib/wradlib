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

        self.img = img

        pixel = np.ones((3, 3)) * 3.5355339059327378
        pixel[1, 1] = 10.
        self.pixel = pixel

        line = np.ones((3, 4)) * 3.5355339059327378
        line[:, 1:3] = 5.0
        line[1, 1:3] = 9.354143466934854
        self.line = line

        spike = np.ones((3, 10)) * 3.0618621784789726
        spike[1] = 4.330127018922194
        spike[:, 0] = 3.1622776601683795, 4.47213595499958, 3.1622776601683795
        spike[:, -1] = 3.1622776601683795, 4.47213595499958, 3.1622776601683795
        self.spike = spike

        rainfield = np.zeros((62, 7))
        rainfield[:, 0:2] = 6.73609679265374
        rainfield[:, -2:] = 6.73609679265374
        rainfield[0:2, :] = 6.73609679265374
        rainfield[-2:, :] = 6.73609679265374
        rainfield[0, :2] = 3.8890872965260113, 5.5
        rainfield[0, -2:] = 5.5, 3.8890872965260113
        rainfield[-1, :2] = 3.8890872965260113, 5.5
        rainfield[-1, -2:] = 5.5, 3.8890872965260113
        rainfield[1, :2] = 5.5, 8.696263565463044
        rainfield[1, -2:] = 8.696263565463044, 5.5
        rainfield[-2, :2] = 5.5, 8.696263565463044
        rainfield[-2, -2:] = 8.696263565463044, 5.5
        self.rainfield = rainfield

    def test_texture(self):
        tex = dp.texture(self.img)
        np.testing.assert_array_equal(tex[1:4, 1:4], self.pixel)
        np.testing.assert_array_equal(tex[4:7, 5:9], self.line)
        np.testing.assert_array_equal(tex[19:22], self.spike)
        np.testing.assert_array_equal(tex[59:121, 1:8], self.rainfield)


class DepolarizationTest(unittest.TestCase):
    def test_depolarization(self):
        zdr = np.linspace(-0.5, 0.5, 10)
        rho = np.linspace(0., 1., 10)

        dr_0 = [-12.719937, -12.746507, -12.766551, -12.779969, -12.786695,
                -12.786695, -12.779969, -12.766551, -12.746507, -12.719937]
        dr_1 = [0., -0.96266, -1.949568, -2.988849, -4.118078, -5.394812,
                -6.921361, -8.919312, -12.067837, -24.806473]

        np.testing.assert_array_almost_equal(dp.depolarization(zdr, 0.9),
                                             dr_0)
        np.testing.assert_array_almost_equal(dp.depolarization(1.0, rho),
                                             dr_1)


if __name__ == '__main__':
    unittest.main()
