#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import os
import numpy as np
import wradlib.util as util
import unittest
import datetime as dt


class HelperFunctionsTest(unittest.TestCase):
    def test__shape_to_size(self):
        self.assertEqual(util._shape_to_size((10, 10, 10)), 10 * 10 * 10)

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
        data = np.arange(0., 30.1, 0.1)
        correct = np.arange(0., 1., 0.01)
        correct = np.concatenate((correct, np.ones(101), correct[::-1]))
        result = util.trapezoid(data, 0., 10., 20., 30.)
        np.testing.assert_array_almost_equal(result, correct, decimal=9)

    def test_prob_round(self):
        np.random.seed(42)
        np.testing.assert_equal(42., util.prob_round(42.4242))
        np.random.seed(44)
        np.testing.assert_equal(43., util.prob_round(42.4242))

    def test_get_wradlib_data_path(self):
        wrl_data_path = os.environ.get('WRADLIB_DATA', None)
        del os.environ['WRADLIB_DATA']
        with self.assertRaises(EnvironmentError):
            util.get_wradlib_data_path()
        filename = 'rainbow/2013070308340000dBuZ.azi'
        os.environ['WRADLIB_DATA'] = os.path.join(wrl_data_path, filename)
        with self.assertRaises(EnvironmentError):
            util.get_wradlib_data_path()
        os.environ['WRADLIB_DATA'] = wrl_data_path
        filename = os.path.join(wrl_data_path, "test.dat")
        with self.assertRaises(EnvironmentError):
            util.get_wradlib_data_file(filename)

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

    def test_import_optional(self):
        m = util.import_optional('math')
        np.testing.assert_equal(m.log10(100), 2.0)
        mod = util.import_optional('h8x')
        with self.assertRaises(AttributeError):
            mod.test()

    def test_maximum_intensity_projection(self):
        angle = 0.0
        elev = 0.0

        filename = util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
        data = np.loadtxt(filename)
        # we need to have meter here for the georef function inside mip
        d1 = np.arange(data.shape[1], dtype=np.float) * 1000
        d2 = np.arange(data.shape[0], dtype=np.float)
        data = np.roll(data, (d2 >= angle).nonzero()[0][0], axis=0)

        # calculate max intensity proj
        util.maximum_intensity_projection(data, r=d1, az=d2,
                                          angle=angle, elev=elev)
        util.maximum_intensity_projection(data, autoext=False)

    def test_roll2d_polar(self):
        filename = util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
        data = np.loadtxt(filename)
        result1 = util.roll2d_polar(data, 1, axis=0)
        result2 = util.roll2d_polar(data, -1, axis=0)
        result3 = util.roll2d_polar(data, 1, axis=1)
        result4 = util.roll2d_polar(data, -1, axis=1)

        np.testing.assert_equal(result1, np.roll(data, 1, axis=0))
        np.testing.assert_equal(result2, np.roll(data, -1, axis=0))
        np.testing.assert_equal(result3[:, 1:],
                                np.roll(data, 1, axis=1)[:, 1:])
        np.testing.assert_equal(result4[:, :-1],
                                np.roll(data, -1, axis=1)[:, :-1])

    def test_medfilt_along_axis(self):
        x = np.arange(10).reshape((2, 5)).astype("f4")
        shouldbe = np.array([[0., 1., 2., 3., 3.],
                             [5., 6., 7., 8., 8.]])
        result = util.medfilt_along_axis(x, 3)
        np.testing.assert_allclose(result, shouldbe)

    def test_gradient_along_axis(self):
        x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
        result = util.gradient_along_axis(x)
        shouldbe = np.array([[1., 11., 2., 4., 6., 7.],
                             [1., 11., 12., 14., 16., 17.]])
        np.testing.assert_allclose(result, shouldbe)

    def test_gradient_from_smoothed(self):
        x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
        result = util.gradient_from_smoothed(x)
        shouldbe = np.array([[1., 11., 2., 1.5, 0., 0.],
                             [1., 11., 12., 6.5, 0., 0.]])
        np.testing.assert_allclose(result, shouldbe)


class TestUtil(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        img = np.zeros((36, 10), dtype=np.float32)
        img[2, 2] = 1  # isolated pixel
        img[5, 6:8] = 1  # line
        img[20, :] = 1  # spike
        img[9:12, 4:7] = 1  # precip field
        # img[15:17,5:7] = np.nan # nodata as nans
        self.img = img

    def test_filter_window_polar(self):
        rscale = 250
        # nrays, nbins = self.img.shape
        # ascale = 2 * np.pi / self.img.shape[0]
        mean = util.filter_window_polar(self.img, 300, "maximum", rscale)
        mean2 = util.filter_window_polar(self.img, 300, "maximum", rscale,
                                         random=True)
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

        correct2 = np.array([[0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
                             [0., 1., 1., 0., 0., 1., 1., 1., 1., 0.],
                             [1., 1., 0., 0., 0., 1., 1., 1., 1., 0.],
                             [1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                             [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                             [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                             [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                             [1., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                             [1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])

        np.testing.assert_array_equal(mean, correct)
        np.testing.assert_array_equal(mean2, correct2)

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
