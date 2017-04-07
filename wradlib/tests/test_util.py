#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import os
import numpy as np
import wradlib.util as util
import unittest
import datetime as dt


class HelperFunctionsTest(unittest.TestCase):
    def test__get_func(self):
        self.assertEqual(util._get_func('arange').__class__,
                         np.arange.__class__)
        self.assertEqual(util._get_func('arange').__module__,
                         np.arange.__module__)
        self.assertRaises(AttributeError, lambda: util._get_func('aranged'))

    def test__shape2size(self):
        self.assertEqual(util._shape2size((10, 10, 10)), 10 * 10 * 10)

    def test__tdelta2seconds(self):
        self.assertEqual(util._tdelta2seconds(dt.datetime(2001, 1, 1, 1) -
                                              dt.datetime(2000, 1, 1)),
                         366 * 24 * 60 * 60 + 3600)
        self.assertEqual(util._tdelta2seconds(dt.datetime(2002, 1, 1, 1) -
                                              dt.datetime(2001, 1, 1)),
                         365 * 24 * 60 * 60 + 3600)

    def test__get_tdelta(self):
        tstart = dt.datetime(2000, 1, 1)
        tend = dt.datetime(2001, 1, 1, 1)
        tstart_str = "2000-01-01 00:00:00"
        tend_str = "2001-01-01 01:00:00"
        self.assertEqual(util._get_tdelta(tstart, tend), tend - tstart)
        self.assertEqual(util._get_tdelta(tstart_str, tend_str), tend - tstart)
        self.assertRaises(ValueError,
                          lambda: util._get_tdelta(tstart_str,
                                                   tend_str + ".00000"))
        self.assertEqual(util._get_tdelta(tstart_str, tend_str, as_secs=True),
                         366 * 24 * 60 * 60 + 3600)

    def test__idvalid(self):
        data = np.array(
            [np.inf, np.nan, -99., 99, -9999., -9999, -10., -5., 0., 5., 10.])
        self.assertTrue(
            np.allclose(util._idvalid(data), np.array([6, 7, 8, 9, 10])))
        self.assertTrue(np.allclose(util._idvalid(data, minval=-5., maxval=5.),
                                    np.array([7, 8, 9])))
        self.assertTrue(
            np.allclose(util._idvalid(data, isinvalid=[-9999], maxval=5.),
                        np.array([2, 6, 7, 8, 9])))

    def test_issequence(self):
        self.assertTrue(util.issequence([0, 1, 2]))
        self.assertFalse(util.issequence(1))
        self.assertFalse(util.issequence('str'))

    def test_trapezoid(self):
        pass

    def test_prob_round(self):
        pass

    def test_get_wradlib_data_path(self):
        wrl_data_path = os.environ.get('WRADLIB_DATA', None)
        del os.environ['WRADLIB_DATA']
        self.assertRaises(EnvironmentError,
                          lambda: util.get_wradlib_data_path())
        filename = 'rainbow/2013070308340000dBuZ.azi'
        os.environ['WRADLIB_DATA'] = os.path.join(wrl_data_path, filename)
        self.assertRaises(EnvironmentError,
                          lambda: util.get_wradlib_data_path())
        os.environ['WRADLIB_DATA'] = wrl_data_path
        filename = os.path.join(wrl_data_path, "test.dat")
        self.assertRaises(EnvironmentError,
                          lambda: util.get_wradlib_data_file(filename))

    def test_from_to(self):
        out = util.from_to("2000-01-01 00:00:00",
                           "2000-01-02 00:00:00",
                           86400)
        shouldbe = [dt.datetime(2000, 1, 1, 0, 0),
                    dt.datetime(2000, 1, 2, 0, 0)]
        self.assertEqual(out, shouldbe)

    def test_calculate_polynomial(self):
        data = np.arange(0, 10, 1)
        w = np.arange(0, 5, 1)
        out = np.array([0, 10, 98, 426, 1252, 2930, 5910, 10738, 18056, 28602])
        poly = util.calculate_polynomial(data, w)
        np.testing.assert_allclose(poly, out, rtol=1e-12)


# -------------------------------------------------------------------------------
# testing the filter helper function
# -------------------------------------------------------------------------------
class TestUtil(unittest.TestCase):
    def setUp(self):
        img = np.zeros((36, 10), dtype=np.float32)
        img[2, 2] = 1  # isolated pixel
        img[5, 6:8] = 1  # line
        img[20, :] = 1  # spike
        img[9:12, 4:7] = 1  # precip field
        # img[15:17,5:7] = np.nan # nodata as nans
        self.img = img

    def test_filter_window_polar(self):
        np.set_printoptions(precision=3)
        rscale = 250
        # nrays, nbins = self.img.shape
        # ascale = 2 * np.pi / self.img.shape[0]
        mean = util.filter_window_polar(self.img, 300, "maximum", rscale)
        correct = np.array([[0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
                            [0., 1., 1., 0., 0., 1., 1., 1., 1., 0.],
                            [1., 1., 0., 0., 0., 1., 1., 1., 1., 0.],
                            [1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                            [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                            [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                            [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                            [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                            [1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])

        self.assertTrue((mean == correct).all())

    def test_half_power_radius(self):
        hpr = util.half_power_radius(np.arange(0, 100000, 10000), 1.0)
        res = np.array([0., 87.266, 174.533, 261.799, 349.066, 436.332,
                        523.599, 610.865, 698.132, 785.398])
        self.assertTrue(np.allclose(hpr, res))

    def test_filter_window_cartesian(self):
        correct = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertTrue(np.allclose(
            util.filter_window_cartesian(self.img, 500., "maximum",
                                         np.array([250., 250])),
            correct))


class FindBboxIndicesTest(unittest.TestCase):
    def setUp(self):
        xarr = np.linspace(500, 1000, num=6)
        yarr = np.linspace(550, 950, num=9)

        gridx, gridy = np.meshgrid(xarr, yarr)

        self.grid = np.dstack((gridx, gridy))
        self.outside = [400, 400, 1100, 1100]
        self.inside1 = [599, 599, 901, 901]
        self.inside2 = [601, 601, 899, 899]

    def test_find_bbox_indices(self):
        bbind = util.find_bbox_indices(self.grid, self.outside)
        self.assertTrue(np.array_equal(bbind, [0, 0, self.grid.shape[1],
                                               self.grid.shape[0]]))

        bbind = util.find_bbox_indices(self.grid, self.inside1)
        self.assertTrue(np.array_equal(bbind, [0, 0, 5, 8]))

        bbind = util.find_bbox_indices(self.grid, self.inside2)
        self.assertTrue(np.array_equal(bbind, [1, 1, 4, 7]))
