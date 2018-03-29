#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.vpr as vpr
import wradlib.georef as georef
import numpy as np


class VPRHelperFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.site = (7.0, 53.0, 100.)
        self.proj = georef.epsg_to_osr(31467)
        self.az = np.arange(0., 360., 2.)
        self.r = np.arange(0, 50000, 1000)
        self.el = 2.5

    def test_out_of_range(self):
        pass

    def test_blindspots(self):
        pass

    def test_volcoords_from_polar(self):
        coords = vpr.volcoords_from_polar(self.site, self.el, self.az, self.r,
                                          self.proj)
        self.assertEqual(coords.shape, (9000, 3))

    def test_volcoords_from_polar_irregular(self):
        # oneazforall, onerange4all, one elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el],
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (9000, 3))

        # oneazforall, onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (18000, 3))

        # onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    [self.az, self.az], self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (18000, 3))

        # oneazforall, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    self.az, [self.r, self.r],
                                                    self.proj)

        self.assertEqual(coords.shape, (18000, 3))

    def test_synthetic_polar_volume(self):
        nbins = [320, 240, 340, 300]
        rscale = [1000, 1000, 500, 500]
        elev = [0.3, 0.4, 3., 4.5]

        xyz = np.array([]).reshape((-1, 3))
        for i, vals in enumerate(zip(nbins, rscale, elev)):
            az = np.arange(0., 360., 2.)
            r = np.arange(0, vals[0] * vals[1], vals[1])
            xyz_ = vpr.volcoords_from_polar(self.site, vals[2],
                                            az, r, self.proj)
            xyz = np.vstack((xyz, xyz_))

        vol = vpr.synthetic_polar_volume(xyz)
        self.assertEqual(vol.shape, (216000,))

    def test_norm_vpr_stats(self):
        vol = np.arange(2 * 3 * 4).astype("f4").reshape((4, 3, 2)) ** 2
        prof = vpr.norm_vpr_stats(vol, 1)
        np.allclose(prof, np.array([0.09343848, 1., 3.0396144, 6.2122827]))

    def test_make_3d_grid(self):
        maxrange = 50000.
        maxalt = 5000.
        horiz_res = 4000.
        vert_res = 1000.
        outxyz, outshape = vpr.make_3d_grid(self.site, self.proj, maxrange,
                                            maxalt, horiz_res, vert_res)
        self.assertEqual(outshape, (6, 26, 26))
        self.assertEqual(outxyz.shape, (4056, 3))


class CartesianVolumeTest(unittest.TestCase):
    def setUp(self):
        # polar grid settings
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
        out = gridder(self.data)
        self.assertEqual(out.shape, (6084,))
        self.assertEqual(len(np.where(np.isnan(out))[0]), 0)

    def test_CAPPI(self):
        gridder = vpr.CAPPI(self.xyz, self.trgxyz, self.trgshape,
                            self.maxrange, self.minelev, self.maxelev)
        out = gridder(self.data)
        self.assertEqual(out.shape, (6084,))
        self.assertEqual(len(np.where(np.isnan(out))[0]), 3512)

    def test_PseudoCAPPI(self):
        # interpolate to Cartesian 3-D volume grid
        gridder = vpr.PseudoCAPPI(self.xyz, self.trgxyz, self.trgshape,
                                  self.maxrange, self.minelev, self.maxelev)
        out = gridder(self.data)
        self.assertEqual(out.shape, (6084,))
        self.assertEqual(len(np.where(np.isnan(out))[0]), 1744)


if __name__ == '__main__':
    unittest.main()
