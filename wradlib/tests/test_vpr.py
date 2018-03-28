#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.vpr as vpr
import wradlib.georef as georef
import numpy as np


class VPRHelperFunctionsTest(unittest.TestCase):
    def setUp(self):
        # polar grid cettings
        self.site = (7.0, 53.0, 100.)
        self.proj = georef.epsg_to_osr(31467)
        self.az = np.arange(0., 360., 2.) + 1.
        self.r = np.arange(0., 50000., 1000.)
        self.elev = np.array([1., 3., 5., 10.])
        # cartesian grid settings
        self.maxrange = 50000.
        self.minelev = 1.
        self.maxelev = 10.
        self.maxalt = 8000.
        self.horiz_res = 4000.
        self.vert_res = 1000.
        self.xyz = vpr.volcoords_from_polar(self.site, self.elev,
                                            self.az, self.r, self.proj)
        self.data = vpr.synthetic_polar_volume(self.xyz)
        self.trgxyz, self.trgshape = vpr.make_3d_grid(self.site, self.proj,
                                                      self.maxrange,
                                                      self.maxalt,
                                                      self.horiz_res,
                                                      self.vert_res)

    def test_out_of_range(self):
        pass

    def test_blindspots(self):
        pass

    def test_volcoords_from_polar(self):
        self.assertEqual(self.xyz.shape, (36000, 3))

    def test_volcoords_from_polar_irregular(self):
        # oneazforall, onerange4all, one elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.elev[0]],
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (9000, 3))

        # oneazforall, onerange4all, all elev
        coords = vpr.volcoords_from_polar_irregular(self.site, self.elev,
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (36000, 3))

        # one az
        coords = vpr.volcoords_from_polar_irregular(self.site, self.elev,
                                                    [self.az[0]], self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (200, 3))

        # one r
        coords = vpr.volcoords_from_polar_irregular(self.site, self.elev,
                                                    self.az, [self.r[0]],
                                                    self.proj)

        self.assertEqual(coords.shape, (720, 3))

    def test_synthetic_polar_volume(self):
        self.assertEqual(self.data.shape, (36000,))

    def test_vpr_interpolator(self):
        pass

    def test_correct_vpr(self):
        pass

    def test_mean_norm_from_vpr(self):
        pass

    def test_norm_vpr_stats(self):
        pass

    def test_make_3d_grid(self):
        self.assertEqual(self.trgshape, (9, 26, 26))
        self.assertEqual(self.trgxyz.shape, (6084, 3))


class CartesianVolumeTest(unittest.TestCase):
    def setUp(self):
        # polar grid cettings
        self.site = (7.0, 53.0, 100.)
        self.proj = georef.epsg_to_osr(31467)
        self.az = np.arange(0., 360., 2.) + 1.
        self.r = np.arange(0., 50000., 1000.)
        self.elev = np.array([1., 3., 5., 10.])
        # cartesian grid settings
        self.maxrange = 50000.
        self.minelev = 1.
        self.maxelev = 10.
        self.maxalt = 8000.
        self.horiz_res = 4000.
        self.vert_res = 1000.
        self.xyz = vpr.volcoords_from_polar(self.site, self.elev,
                                            self.az, self.r, self.proj)
        self.data = vpr.synthetic_polar_volume(self.xyz)
        self.trgxyz, self.trgshape = vpr.make_3d_grid(self.site, self.proj,
                                                      self.maxrange,
                                                      self.maxalt,
                                                      self.horiz_res,
                                                      self.vert_res)

    def test_CartesianVolume(self):
        gridder = vpr.CartesianVolume(self.xyz, self.trgxyz, self.trgshape,
                                      self.maxrange, self.minelev,
                                      self.maxelev)
        out = np.ma.masked_invalid(gridder(self.data).reshape(self.trgshape))
        self.assertEqual(out.shape, (9, 26, 26))

    def test_CAPPI(self):
        gridder = vpr.CAPPI(self.xyz, self.trgxyz, self.trgshape,
                            self.maxrange, self.minelev, self.maxelev)
        out = np.ma.masked_invalid(gridder(self.data).reshape(self.trgshape))
        self.assertEqual(out.shape, (9, 26, 26))

    def test_PseudoCAPPI(self):
        # interpolate to Cartesian 3-D volume grid
        gridder = vpr.PseudoCAPPI(self.xyz, self.trgxyz, self.trgshape,
                                  self.maxrange, self.minelev, self.maxelev)
        out = np.ma.masked_invalid(gridder(self.data).reshape(self.trgshape))
        self.assertEqual(out.shape, (9, 26, 26))


if __name__ == '__main__':
    unittest.main()
