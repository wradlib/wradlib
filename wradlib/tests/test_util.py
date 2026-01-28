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

from . import get_wradlib_data_file, requires_gdal


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


def test_get_wradlib_data():
    data_path = util._get_wradlib_data_path()
    filename = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
    util._get_wradlib_data_file(filename)
    os.environ["WRADLIB_DATA"] = str(data_path)
    assert util._get_wradlib_data_path() == data_path
    util._get_wradlib_data_file(filename)


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


def test_roll2d_polar():
    filename = get_wradlib_data_file("misc/polar_dBZ_tur.gz")
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
        img_arr = xr.DataArray(
            img,
            dims=["azimuth", "range"],
            coords=[
                np.linspace(0, 360, 36, endpoint=False),
                np.linspace(0, 2500, 10, endpoint=False),
            ],
        )
        img_rect = xr.DataArray(
            img,
            dims=["y", "x"],
            coords=[
                np.linspace(0, 9000, 36, endpoint=False),
                np.linspace(0, 2500, 10, endpoint=False),
            ],
        )

    yield TestUtil


def test_filter_window_polar(udata):
    np.random.seed(42)
    rscale = 250
    mean = util.filter_window_polar(udata.img.copy(), 300, "maximum", rscale)
    mean2 = util.filter_window_polar(
        udata.img.copy(), 300, "maximum", rscale, random=True
    )

    np.random.seed(42)
    meanx = udata.img_arr.copy(deep=True).wrl.util.filter_window_polar(
        wsize=300, fun="maximum"
    )
    meanx2 = udata.img_arr.copy(deep=True).wrl.util.filter_window_polar(
        wsize=300, fun="maximum", random=True
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
    np.testing.assert_array_equal(meanx.values, correct)
    np.testing.assert_array_equal(meanx2.values, correct2)


def test_half_power_radius():
    rdata = np.arange(0, 100000, 10000)
    rng = xr.DataArray(rdata, dims=["range"], coords=[rdata])
    rng.name = "test"
    hpr = util.half_power_radius(rdata, 1.0)
    hprx = rng.wrl.util.half_power_radius(1.0)
    hprx2 = rng.to_dataset().wrl.util.half_power_radius(1.0)
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
    assert np.allclose(hprx.values, res)
    assert np.allclose(hprx2.values, res)


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

    assert np.allclose(
        udata.img_rect.wrl.util.filter_window_cartesian(wsize=500.0, fun="maximum"),
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
def test_cross_section_ppi():
    file = get_wradlib_data_file("hdf5/71_20181220_061228.pvol.h5")
    # only load DBZH for testing
    keep_vars = ["DBZH", "sweep_mode", "sweep_number", "prt_mode", "follow_mode"]
    # load all sweeps manually and merge them
    sweeps = []
    for sn in np.arange(14):
        sweep = xr.open_dataset(
            file, engine="odim", group="sweep_" + str(sn)
        ).set_coords("sweep_fixed_angle")
        sweep = sweep[keep_vars]
        sweeps.append(sweep)
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

    assert rec_rhi.sizes["azimuth"] == 3

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


@pytest.fixture
def texture_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestTexture:
        img = np.zeros((360, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[60:120, 2:7] = 11  # precip field

        pixel = np.ones((3, 3)) * 3.5355339059327378
        pixel[1, 1] = 10.0

        line = np.ones((3, 4)) * 3.5355339059327378
        line[:, 1:3] = 5.0
        line[1, 1:3] = 9.354143466934854

        spike = np.ones((3, 10)) * 3.0618621784789726
        spike[1] = 4.330127018922194
        spike[:, 0] = 3.1622776601683795, 4.47213595499958, 3.1622776601683795
        spike[:, -1] = 3.1622776601683795, 4.47213595499958, 3.1622776601683795

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

    yield TestTexture


def test_texture(texture_data):
    tex = util.texture(texture_data.img)
    np.testing.assert_array_equal(tex[1:4, 1:4], texture_data.pixel)
    np.testing.assert_array_equal(tex[4:7, 5:9], texture_data.line)
    np.testing.assert_array_equal(tex[19:22], texture_data.spike)
    np.testing.assert_array_equal(tex[59:121, 1:8], texture_data.rainfield)


def test_texture_xarray(gamic_swp):
    gamic_swp.wrl.dp.texture()
    gamic_swp.PHIDP.wrl.dp.texture()
