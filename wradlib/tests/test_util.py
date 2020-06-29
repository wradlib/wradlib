#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import datetime as dt
import os

import deprecation
import numpy as np
import pytest

from wradlib import util

from . import requires_data


class TestHelperFunctions:
    def test__shape_to_size(self):
        assert util._shape_to_size((10, 10, 10)) == 10 * 10 * 10

    def test__idvalid(self):
        data = np.array(
            [np.inf, np.nan, -99.0, 99, -9999.0, -9999, -10.0, -5.0, 0.0, 5.0, 10.0]
        )
        assert np.allclose(util._idvalid(data), np.array([6, 7, 8, 9, 10]))
        assert np.allclose(
            util._idvalid(data, minval=-5.0, maxval=5.0), np.array([7, 8, 9])
        )
        assert np.allclose(
            util._idvalid(data, isinvalid=[-9999], maxval=5.0),
            np.array([2, 6, 7, 8, 9]),
        )

    def test_issequence(self):
        assert util.issequence([0, 1, 2])
        assert not util.issequence(1)
        assert not util.issequence("str")

    def test_trapezoid(self):
        data = np.arange(0.0, 30.1, 0.1)
        correct = np.arange(0.0, 1.0, 0.01)
        correct = np.concatenate((correct, np.ones(101), correct[::-1]))
        result = util.trapezoid(data, 0.0, 10.0, 20.0, 30.0)
        np.testing.assert_array_almost_equal(result, correct, decimal=9)

    def test_prob_round(self):
        np.random.seed(42)
        np.testing.assert_equal(42.0, util.prob_round(42.4242))
        np.random.seed(44)
        np.testing.assert_equal(43.0, util.prob_round(42.4242))

    def test_get_wradlib_data_path(self):
        wrl_data_path = os.environ.get("WRADLIB_DATA", None)
        del os.environ["WRADLIB_DATA"]
        with pytest.raises(EnvironmentError):
            util.get_wradlib_data_path()
        if wrl_data_path is not None:
            os.environ["WRADLIB_DATA"] = wrl_data_path

    @requires_data
    def test_get_wradlib_data_path_requires(self):
        filename = os.path.join(util.get_wradlib_data_path(), "test.dat")
        with pytest.raises(EnvironmentError):
            util.get_wradlib_data_file(filename)

    def test_from_to(self):
        out = util.from_to("2000-01-01 00:00:00", "2000-01-02 00:00:00", 86400)
        shouldbe = [dt.datetime(2000, 1, 1, 0, 0), dt.datetime(2000, 1, 2, 0, 0)]
        assert out == shouldbe

    def test_calculate_polynomial(self):
        data = np.arange(0, 10, 1)
        w = np.arange(0, 5, 1)
        out = np.array([0, 10, 98, 426, 1252, 2930, 5910, 10738, 18056, 28602])
        poly = util.calculate_polynomial(data, w)
        np.testing.assert_allclose(poly, out, rtol=1e-12)

    def test_import_optional(self):
        m = util.import_optional("math")
        np.testing.assert_equal(m.log10(100), 2.0)
        mod = util.import_optional("h8x")
        with pytest.raises(AttributeError):
            mod.test()

    @requires_data
    @deprecation.fail_if_not_removed
    def test_maximum_intensity_projection(self):
        angle = 0.0
        elev = 0.0

        filename = util.get_wradlib_data_file("misc/polar_dBZ_tur.gz")
        data = np.loadtxt(filename)
        # we need to have meter here for the georef function inside mip
        d1 = np.arange(data.shape[1], dtype=np.float) * 1000
        d2 = np.arange(data.shape[0], dtype=np.float)
        data = np.roll(data, (d2 >= angle).nonzero()[0][0], axis=0)

        # calculate max intensity proj
        util.maximum_intensity_projection(data, r=d1, az=d2, angle=angle, elev=elev)
        util.maximum_intensity_projection(data, autoext=False)

    @requires_data
    def test_roll2d_polar(self):
        filename = util.get_wradlib_data_file("misc/polar_dBZ_tur.gz")
        data = np.loadtxt(filename)
        result1 = util.roll2d_polar(data, 1, axis=0)
        result2 = util.roll2d_polar(data, -1, axis=0)
        result3 = util.roll2d_polar(data, 1, axis=1)
        result4 = util.roll2d_polar(data, -1, axis=1)

        np.testing.assert_equal(result1, np.roll(data, 1, axis=0))
        np.testing.assert_equal(result2, np.roll(data, -1, axis=0))
        np.testing.assert_equal(result3[:, 1:], np.roll(data, 1, axis=1)[:, 1:])
        np.testing.assert_equal(result4[:, :-1], np.roll(data, -1, axis=1)[:, :-1])

    def test_medfilt_along_axis(self):
        x = np.arange(10).reshape((2, 5)).astype("f4")
        shouldbe = np.array([[0.0, 1.0, 2.0, 3.0, 3.0], [5.0, 6.0, 7.0, 8.0, 8.0]])
        result = util.medfilt_along_axis(x, 3)
        np.testing.assert_allclose(result, shouldbe)

    def test_gradient_along_axis(self):
        x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
        result = util.gradient_along_axis(x)
        shouldbe = np.array([[1.0, 2.0, 4.0, 6.0, 7.0], [11.0, 12.0, 14.0, 16.0, 17.0]])
        np.testing.assert_allclose(result, shouldbe)

    def test_gradient_from_smoothed(self):
        x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
        result = util.gradient_from_smoothed(x)
        shouldbe = np.array([[1.0, 2.0, 1.5, 0.0, 0.0], [11.0, 12.0, 6.5, 0.0, 0.0]])
        np.testing.assert_allclose(result, shouldbe)


class TestUtil:
    img = np.zeros((36, 10), dtype=np.float64)
    img[2, 2] = 1  # isolated pixel
    img[5, 6:8] = 1  # line
    img[20, :] = 1  # spike
    img[9:12, 4:7] = 1  # precip field
    # img[15:17,5:7] = np.nan # nodata as nans

    def test_filter_window_polar(self):
        np.random.seed(42)
        rscale = 250
        # nrays, nbins = self.img.shape
        # ascale = 2 * np.pi / self.img.shape[0]
        mean = util.filter_window_polar(self.img.copy(), 300, "maximum", rscale)
        mean2 = util.filter_window_polar(
            self.img.copy(), 300, "maximum", rscale, random=True
        )
        correct = np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        correct2 = np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(mean, correct)
        np.testing.assert_array_equal(mean2, correct2)

    def test_half_power_radius(self):
        hpr = util.half_power_radius(np.arange(0, 100000, 10000), 1.0)
        res = np.array(
            [
                0.0,
                87.266,
                174.533,
                261.799,
                349.066,
                436.332,
                523.599,
                610.865,
                698.132,
                785.398,
            ]
        )
        assert np.allclose(hpr, res)

    def test_filter_window_cartesian(self):
        correct = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert np.allclose(
            util.filter_window_cartesian(
                self.img, 500.0, "maximum", np.array([250.0, 250])
            ),
            correct,
        )


class TestFindBboxIndices:
    xarr = np.linspace(500, 1000, num=6)
    yarr = np.linspace(550, 950, num=9)

    gridx, gridy = np.meshgrid(xarr, yarr)

    grid = np.dstack((gridx, gridy))
    outside = [400, 400, 1100, 1100]
    inside1 = [599, 599, 901, 901]
    inside2 = [601, 601, 899, 899]

    def test_find_bbox_indices(self):
        bbind = util.find_bbox_indices(self.grid, self.outside)
        assert np.array_equal(bbind, [0, 0, self.grid.shape[1], self.grid.shape[0]])

        bbind = util.find_bbox_indices(self.grid, self.inside1)
        assert np.array_equal(bbind, [0, 0, 5, 8])

        bbind = util.find_bbox_indices(self.grid, self.inside2)
        assert np.array_equal(bbind, [1, 1, 4, 7])
