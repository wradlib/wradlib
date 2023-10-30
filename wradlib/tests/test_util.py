#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from wradlib import util

from . import requires_data, requires_gdal


def test__shape_to_size():
    assert util._shape_to_size((10, 10, 10)) == 10 * 10 * 10


def test__idvalid():
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


def test_issequence():
    assert util.issequence([0, 1, 2])
    assert not util.issequence(1)
    assert not util.issequence("str")


def test_trapezoid():
    data = np.arange(0.0, 30.1, 0.1)
    correct = np.arange(0.0, 1.0, 0.01)
    correct = np.concatenate((correct, np.ones(101), correct[::-1]))
    result = util.trapezoid(data, 0.0, 10.0, 20.0, 30.0)
    np.testing.assert_array_almost_equal(result, correct, decimal=9)


def test_prob_round():
    np.random.seed(42)
    np.testing.assert_equal(42.0, util.prob_round(42.4242))
    np.random.seed(44)
    np.testing.assert_equal(43.0, util.prob_round(42.4242))


def test_get_wradlib_data_path():
    wrl_data_path = os.environ.get("WRADLIB_DATA", None)
    del os.environ["WRADLIB_DATA"]
    with pytest.raises(EnvironmentError):
        util.get_wradlib_data_path()
    if wrl_data_path is not None:
        os.environ["WRADLIB_DATA"] = wrl_data_path


@requires_data
def test_get_wradlib_data_path_requires():
    filename = os.path.join(util.get_wradlib_data_path(), "test.dat")
    with pytest.raises((EnvironmentError, ValueError)):
        util.get_wradlib_data_file(filename)


def test_from_to():
    out = util.from_to("2000-01-01 00:00:00", "2000-01-02 00:00:00", 86400)
    shouldbe = [dt.datetime(2000, 1, 1, 0, 0), dt.datetime(2000, 1, 2, 0, 0)]
    assert out == shouldbe


def test_calculate_polynomial():
    data = np.arange(0, 10, 1)
    w = np.arange(0, 5, 1)
    out = np.array([0, 10, 98, 426, 1252, 2930, 5910, 10738, 18056, 28602])
    poly = util.calculate_polynomial(data, w)
    np.testing.assert_allclose(poly, out, rtol=1e-12)


def test_import_optional():
    m = util.import_optional("math")
    np.testing.assert_equal(m.log10(100), 2.0)
    mod = util.import_optional("h8x")
    with pytest.raises(AttributeError):
        mod.test()


@requires_data
def test_roll2d_polar():
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


def test_medfilt_along_axis():
    x = np.arange(10).reshape((2, 5)).astype("f4")
    shouldbe = np.array([[0.0, 1.0, 2.0, 3.0, 3.0], [5.0, 6.0, 7.0, 8.0, 8.0]])
    result = util.medfilt_along_axis(x, 3)
    np.testing.assert_allclose(result, shouldbe)


def test_gradient_along_axis():
    x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
    result = util.gradient_along_axis(x)
    shouldbe = np.array([[1.0, 2.0, 4.0, 6.0, 7.0], [11.0, 12.0, 14.0, 16.0, 17.0]])
    np.testing.assert_allclose(result, shouldbe)


def test_gradient_from_smoothed():
    x = np.arange(10).reshape((2, 5)).astype("f4") ** 2
    result = util.gradient_from_smoothed(x)
    shouldbe = np.array([[1.0, 2.0, 1.5, 0.0, 0.0], [11.0, 12.0, 6.5, 0.0, 0.0]])
    np.testing.assert_allclose(result, shouldbe)


@pytest.fixture
def udata():
    @dataclass(init=False, repr=False, eq=False)
    class TestUtil:
        img = np.zeros((36, 10), dtype=np.float64)
        img[2, 2] = 1  # isolated pixel
        img[5, 6:8] = 1  # line
        img[20, :] = 1  # spike
        img[9:12, 4:7] = 1  # precip field

    yield TestUtil


def test_filter_window_polar(udata):
    np.random.seed(42)
    rscale = 250
    mean = util.filter_window_polar(udata.img.copy(), 300, "maximum", rscale)
    mean2 = util.filter_window_polar(
        udata.img.copy(), 300, "maximum", rscale, random=True
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


def test_half_power_radius():
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


def test_filter_window_cartesian(udata):
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
            udata.img, 500.0, "maximum", np.array([250.0, 250])
        ),
        correct,
    )


@pytest.fixture
def bb_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestFindBboxIndices:
        xarr = np.linspace(500, 1000, num=6)
        yarr = np.linspace(550, 950, num=9)

        gridx, gridy = np.meshgrid(xarr, yarr)

        grid = np.dstack((gridx, gridy))
        outside = [400, 400, 1100, 1100]
        inside1 = [599, 599, 901, 901]
        inside2 = [601, 601, 899, 899]

    yield TestFindBboxIndices


def test_find_bbox_indices(bb_data):
    bbind = util.find_bbox_indices(bb_data.grid, bb_data.outside)
    assert np.array_equal(bbind, [0, 0, bb_data.grid.shape[1], bb_data.grid.shape[0]])

    bbind = util.find_bbox_indices(bb_data.grid, bb_data.inside1)
    assert np.array_equal(bbind, [0, 0, 5, 8])

    bbind = util.find_bbox_indices(bb_data.grid, bb_data.inside2)
    assert np.array_equal(bbind, [1, 1, 4, 7])


@requires_gdal
@requires_data
def test_cross_section_ppi():
    file = util.get_wradlib_data_file("hdf5/71_20181220_061228.pvol.h5")
    # load all sweeps manually and merge them
    sweeps = []
    for sn in np.arange(14):
        sweeps.append(xr.open_dataset(file, engine="odim", group="sweep_" + str(sn)))
        sweeps[-1].coords["azimuth"] = (
            sweeps[-1].coords["azimuth"].round(1)
        )  # round the azimuths to avoid slight differences
    vol = xr.concat(sweeps, dim="sweep_fixed_angle")

    # Pass meta variables to coords to avoid some issues
    vol = vol.set_coords(("sweep_mode", "sweep_number", "prt_mode", "follow_mode"))

    # Reduce coordinates so the georeferencing works
    vol["elevation"] = vol["elevation"].median("azimuth")
    vol["time"] = vol["time"].min("azimuth")
    vol["sweep_mode"] = vol["sweep_mode"].min()

    # Test extract single azimuth
    azimuth = 120.5
    rec_rhi = util.cross_section_ppi(vol, azimuth, method="nearest")

    assert "azimuth" not in rec_rhi.dims

    # Test extract multiple azimuths
    azimuth = [90.5, 120.5, 175.5]
    rec_rhi = util.cross_section_ppi(vol, azimuth, method="nearest")

    assert rec_rhi.dims["azimuth"] == 3

    # Test extract along line
    p1 = (10000, -40000)
    p2 = (20000, 30000)
    rec_rhi = util.cross_section_ppi(vol, (p1, p2), method="nearest")

    assert "xy" in rec_rhi.coords

    # Test custom parameters
    azimuth = 90  # Example azimuth angle
    custom_kwargs = {
        "method": "nearest",
        "bw": 0.9,
        "npl": 2000,
    }
    rec_rhi = util.cross_section_ppi(vol, azimuth, **custom_kwargs)

    assert "azimuth" not in rec_rhi.dims
    assert rec_rhi.DBZH.all("range").any()
