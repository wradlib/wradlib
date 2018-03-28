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
        self.az = np.arange(0., 360., 1.)
        self.r = np.arange(0, 100000, 1000)
        self.el = 2.5

    def test_out_of_range(self):
        pass

    def test_blindspots(self):
        pass

    def test_volcoords_from_polar(self):
        coords = vpr.volcoords_from_polar(self.site, self.el, self.az, self.r,
                                          self.proj)
        self.assertEqual(coords.shape, (36000, 3))

    def test_volcoords_from_polar_irregular(self):
        # oneazforall, onerange4all, one elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el],
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (36000, 3))

        # oneazforall, onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    self.az, self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (72000, 3))

        # onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    [self.az, self.az], self.r,
                                                    self.proj)
        self.assertEqual(coords.shape, (72000, 3))

        # oneazforall, two elev
        coords = vpr.volcoords_from_polar_irregular(self.site, [self.el, 5.0],
                                                    self.az, [self.r, self.r],
                                                    self.proj)

        self.assertEqual(coords.shape, (72000, 3))

    def test_synthetic_polar_volume(self):
        nbins = [320, 240, 240, 240, 240, 340, 340, 300, 300,
                 240, 240, 240, 240, 240]
        rscale = [1000, 1000, 1000, 1000, 1000, 500, 500, 500, 500, 500, 500,
                  500, 500, 500]
        elev = [0.3, 0.4, 0.8, 1.1, 2., 3., 4.5, 6., 8., 10.,
                12., 15., 20., 25.]

        xyz = np.array([]).reshape((-1, 3))
        for i, vals in enumerate(zip(nbins, rscale, elev)):
            az = np.arange(0., 360., 360. / 360)
            r = np.arange(0, vals[0] * vals[1], vals[1])
            xyz_ = vpr.volcoords_from_polar(self.site, vals[2],
                                            az, r, self.proj)
            xyz = np.vstack((xyz, xyz_))

        vol = vpr.synthetic_polar_volume(xyz)
        self.assertEqual(vol.shape, (1353600,))

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
        vpr.make_3d_grid(self.site, self.proj, maxrange, maxalt, horiz_res,
                         vert_res)
        pass


class CartesianVolumeTest(unittest.TestCase):
    def setUp(self):
        self.site = (7.0, 53.0, 100.)
        self.proj = georef.epsg_to_osr(31467)
        self.az = np.arange(0., 360., 2.) + 1.
        self.r = np.arange(0., 50000., 1000.)
        self.elev = np.array([1., 3., 5., 10.])
        self.xyz = vpr.volcoords_from_polar(self.site, self.elev,
                                            self.az, self.r, self.proj)
        self.data = vpr.synthetic_polar_volume(self.xyz)

    def test_CartesianVolume(self):
        pass

    def test_CAPPI(self):
        maxrange = 50000.
        minelev = 1.
        maxelev = 10.
        maxalt = 8000.
        horiz_res = 4000.
        vert_res = 1000.
        trgxyz, trgshape = vpr.make_3D_grid(self.site, self.proj,
                                            maxrange, maxalt,
                                            horiz_res, vert_res)
        # interpolate to Cartesian 3-D volume grid
        gridder = vpr.CAPPI(self.xyz, trgxyz, trgshape, maxrange,
                            minelev, maxelev)
        out = np.ma.masked_invalid(gridder(self.data).reshape(trgshape))
        self.assertEqual(out.shape, (9, 26, 26))

    def test_PseudoCAPPI(self):
        maxrange = 50000.
        minelev = 1.
        maxelev = 10.
        maxalt = 8000.
        horiz_res = 4000.
        vert_res = 1000.
        trgxyz, trgshape = vpr.make_3D_grid(self.site, self.proj,
                                            maxrange, maxalt,
                                            horiz_res, vert_res)
        # interpolate to Cartesian 3-D volume grid
        gridder = vpr.PseudoCAPPI(self.xyz, trgxyz, trgshape, maxrange,
                                  minelev, maxelev)
        out = np.ma.masked_invalid(gridder(self.data).reshape(trgshape))
        self.assertEqual(out.shape, (9, 26, 26))


if __name__ == '__main__':
    unittest.main()
