#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.vpr as vpr
import wradlib.georef as georef
import numpy as np


class VPRHelperFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.site = (7.0, 53.0, 100.)
        self.proj = georef.epsg_to_osr(31467)
        self.az = np.arange(0., 360., 1.)
        self.r = np.arange(0, 100000, 1000)
        self.el = 2.5
        self.coords = vpr.volcoords_from_polar(self.site, self.el, self.az,
                                               self.r, self.proj)

    def test_out_of_range(self):
        pass

    def test_blindspots(self):
        pass

    def test_volcoords_from_polar(self):
        coords = vpr.volcoords_from_polar(self.site, self.el, self.az, self.r,  # noqa
                                          self.proj)
        pass

    def test_volcoords_from_polar_irregular(self):
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],  # noqa
                                                    self.az, self.r,
                                                    self.proj)
        pass

    def test_synthetic_polar_volume(self):
        vol = vpr.synthetic_polar_volume(self.coords)  # noqa
        pass

    def test_vpr_interpolator(self):
        pass

    def test_correct_vpr(self):
        pass

    def test_mean_norm_from_vpr(self):
        pass

    def test_norm_vpr_stats(self):
        pass

    def test_make_3D_grid(self):
        maxrange = 200000.
        maxalt = 5000.
        horiz_res = 2000.
        vert_res = 250.
        vpr.make_3D_grid(self.site, self.proj, maxrange, maxalt, horiz_res,
                         vert_res)
        pass


class CartesianVolumeTest(unittest.TestCase):
    def test_CartesianVolume(self):
        pass

    def test_CAPPI(self):
        pass

    def test_PseudoCAPPI(self):
        pass


if __name__ == '__main__':
    unittest.main()
