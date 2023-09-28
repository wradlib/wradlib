#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import georef, ipol

from . import requires_gdal


@pytest.fixture
def ipol_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestInterpolation:
        # Kriging Variables
        src = np.array([[0.0, 0.0], [4.0, 0]])
        trg = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0], [4.0, 0]])
        src_d = np.array([0.0, 1.0])
        trg_d = np.array([0.0, 1.0, 2.0, 3.0])
        vals = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        # Need to use different test data because Linear requires more points
        # depending on their spatial constellation (in order to create a
        # convex hull)
        src_lin = np.array([[0.0, 0.0], [4.0, 0], [1.0, 1.0]])
        trg_lin = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0], [4.0, 0]])
        vals_lin = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    yield TestInterpolation


def test_parse_covariogram():
    cov_model = "1.0 Exp(10.5) + 2.3 Sph(20.4) + 5.0 Nug(0.)"
    h = 5.0
    c = ipol.parse_covariogram(cov_model)
    ci = sum(
        [
            ipol.cov_exp(h, 1.0, 10.5),
            ipol.cov_sph(h, 2.3, 20.4),
            ipol.cov_nug(h, 5.0, 0.0),
        ]
    )
    assert c(h) == ci


def test_cov_lin():
    assert np.allclose(ipol.cov_lin([0.0, 5.0, 10.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(
        ipol.cov_lin([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 1.0, 0.0]),
    )


def test_cov_sph():
    assert np.allclose(ipol.cov_sph([0.0, 5.0, 10.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(
        ipol.cov_sph([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 0.625, 0.0]),
    )


def test_cov_exp():
    assert np.allclose(
        ipol.cov_exp([0.0, 5.0, 10.0]),
        np.array([1.0, 6.73794700e-03, 4.53999298e-05]),
    )
    assert np.allclose(
        ipol.cov_exp([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 1.21306132, 0.73575888]),
    )


def test_cov_pow():
    assert np.allclose(ipol.cov_pow([0.0, 5.0, 10.0]), np.array([1.0, -4.0, -9.0]))
    assert np.allclose(
        ipol.cov_pow([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.00000000e00, -9.76562300e06, -1.00000000e10]),
    )


def test_cov_mat():
    assert np.allclose(
        ipol.cov_mat([0.0, 5.0, 10.0]),
        np.array([1.00000000e00, 8.49325705e-04, 7.21354153e-07]),
    )
    assert np.allclose(
        ipol.cov_mat([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 0.98613738, 0.48623347]),
    )
    assert np.allclose(
        ipol.cov_mat([0.0, 5.0, 10.0], sill=2.0, rng=10.0, shp=0.25),
        np.array([2.0, 0.74916629, 0.39961004]),
    )


def test_cov_gau():
    assert np.allclose(
        ipol.cov_gau([0.0, 5.0, 10.0]),
        np.array([1.00000000e00, 1.38879439e-11, 3.72007598e-44]),
    )
    assert np.allclose(
        ipol.cov_gau([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 1.55760157, 0.73575888]),
    )


def test_cov_cau():
    assert np.allclose(
        ipol.cov_cau([0.0, 5.0, 10.0]), np.array([1.0, 0.16666667, 0.09090909])
    )
    assert np.allclose(
        ipol.cov_cau([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
        np.array([2.0, 1.33333333, 1.0]),
    )
    assert np.allclose(
        ipol.cov_cau([0.0, 5.0, 10.0], sill=2.0, rng=10.0, alpha=0.5),
        np.array([2.0, 0.6862915, 0.5]),
    )
    assert np.allclose(
        ipol.cov_cau([0.0, 5.0, 10.0], sill=2.0, rng=10.0, alpha=0.5, beta=1.5),
        np.array([2.0, 0.40202025, 0.25]),
    )


def test_Nearest_1(ipol_data):
    """testing the basic behaviour of the Idw class"""
    ip = ipol.Nearest(ipol_data.src, ipol_data.trg)
    # input more than one dataset
    res = ip(ipol_data.vals)
    assert np.allclose(
        res,
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    )
    # input only one flat array
    res = ip(ipol_data.vals[:, 2])
    assert np.allclose(res, np.array([3.0, 3.0, 3.0, 1.0]))


def test_Idw_1(ipol_data):
    """testing the basic behaviour of the Idw class"""
    ip = ipol.Idw(ipol_data.src, ipol_data.trg)
    # input more than one dataset
    res = ip(ipol_data.vals)
    assert np.allclose(
        res,
        np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.2, 2.0, 2.8], [3.0, 2.0, 1.0]]),
    )
    # input only one flat array
    res = ip(ipol_data.vals[:, 2])
    assert np.allclose(res, np.array([3.0, 2.0, 2.8, 1.0]))


def test_Linear_1(ipol_data):
    """testing the basic behaviour of the Linear class"""
    ip = ipol.Linear(ipol_data.src_lin, ipol_data.trg_lin)
    # input more than one dataset
    res = ip(ipol_data.vals_lin)
    assert np.allclose(
        res,
        np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.5, 2.0, 2.5], [3.0, 2.0, 1.0]]),
    )
    # input only one flat array
    res = ip(ipol_data.vals_lin[:, 2])
    assert np.allclose(res, np.array([3.0, 2.0, 2.5, 1.0]))


def test_OrdinaryKriging_1(ipol_data):
    """testing the basic behaviour of the OrdinaryKriging class"""

    ip = ipol.OrdinaryKriging(ipol_data.src, ipol_data.trg, "1.0 Lin(2.0)")
    # input more than one dataset
    res = ip(ipol_data.vals)
    assert np.all(
        res
        == np.array(
            [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.5, 2.0, 2.5], [3.0, 2.0, 1.0]]
        )
    )
    # input only one flat array
    res = ip(ipol_data.vals[:, 2])
    assert np.allclose(res, np.array([3.0, 2.0, 2.5, 1.0]))


def test_ExternalDriftKriging_1(ipol_data):
    """testing the basic behaviour of the ExternalDriftKriging class
    with drift terms constant over multiple fields"""

    ip = ipol.ExternalDriftKriging(
        ipol_data.src,
        ipol_data.trg,
        "1.0 Lin(2.0)",
        src_drift=ipol_data.src_d,
        trg_drift=ipol_data.trg_d,
    )

    # input more than one dataset
    res = ip(ipol_data.vals)
    assert np.all(
        res
        == np.array(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 2.0, -1.0], [7.0, 2.0, -3.0]]
        )
    )
    # input only one flat array
    res = ip(ipol_data.vals[:, 2])
    assert np.allclose(res, np.array([3.0, 1.0, -1.0, -3.0]))


def test_ExternalDriftKriging_2(ipol_data):
    """testing the basic behaviour of the ExternalDriftKriging class
    with drift terms varying over multiple fields"""
    src_d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    trg_d = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )

    ip = ipol.ExternalDriftKriging(
        ipol_data.src, ipol_data.trg, "1.0 Lin(2.0)", src_drift=src_d, trg_drift=trg_d
    )

    res = ip(ipol_data.vals)
    assert np.all(
        res
        == np.array(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 2.0, -1.0], [7.0, 2.0, -3.0]]
        )
    )
    # input only one flat array
    res = ip(ipol_data.vals[:, 2], src_drift=src_d[:, 2], trg_drift=trg_d[:, 2])
    assert np.allclose(res, np.array([3.0, 1.0, -1.0, -3.0]))


def test_ExternalDriftKriging_3(ipol_data):
    """testing the basic behaviour of the ExternalDriftKriging class
    with missing drift terms"""
    ip = ipol.ExternalDriftKriging(
        ipol_data.src, ipol_data.trg, "1.0 Lin(2.0)", src_drift=None, trg_drift=None
    )

    with pytest.raises(ValueError):
        ip(ipol_data.vals)


def test_MissingErrors(ipol_data):
    with pytest.raises(ipol.MissingSourcesError):
        ipol.Nearest(np.array([]), ipol_data.trg)
    with pytest.raises(ipol.MissingTargetsError):
        ipol.Nearest(ipol_data.src, np.array([]))
    with pytest.raises(ipol.MissingSourcesError):
        ipol.Idw(np.array([]), ipol_data.trg)
    with pytest.raises(ipol.MissingTargetsError):
        ipol.Idw(ipol_data.src, np.array([]))
    with pytest.raises(ipol.MissingSourcesError):
        ipol.Linear(np.array([]), ipol_data.trg)
    with pytest.raises(ipol.MissingTargetsError):
        ipol.Linear(ipol_data.src, np.array([]))
    with pytest.raises(ipol.MissingSourcesError):
        ipol.OrdinaryKriging(np.array([]), ipol_data.trg)
    with pytest.raises(ipol.MissingTargetsError):
        ipol.OrdinaryKriging(ipol_data.src, np.array([]))
    with pytest.raises(ipol.MissingSourcesError):
        ipol.ExternalDriftKriging(np.array([]), ipol_data.trg)
    with pytest.raises(ipol.MissingTargetsError):
        ipol.ExternalDriftKriging(ipol_data.src, np.array([]))


def test_nnearest_warning(ipol_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ipol.Idw(ipol_data.src, ipol_data.trg, nnearest=len(ipol_data.src) + 1)
        ipol.OrdinaryKriging(
            ipol_data.src, ipol_data.trg, nnearest=len(ipol_data.src) + 1
        )
        ipol.ExternalDriftKriging(
            ipol_data.src, ipol_data.trg, nnearest=len(ipol_data.src) + 1
        )
        for item in w:
            assert issubclass(item.category, UserWarning)
            assert "nnearest" in str(item.message)


def test_IpolBase(ipol_data):
    """testing the basic behaviour of the base class"""

    ip = ipol.IpolBase(ipol_data.src, ipol_data.trg)
    res = ip(ipol_data.vals)
    assert res is None

    # Check behaviour if args are passed as lists
    src = [ipol_data.src[:, 0], ipol_data.src[:, 1]]
    trg = [ipol_data.trg[:, 0], ipol_data.trg[:, 1]]
    ip = ipol.IpolBase(src, trg)
    assert len(ipol_data.src) == ip.numsources

    # Check behaviour if dimension is > 2
    ip = ipol.IpolBase(ipol_data.src, ipol_data.trg)
    with pytest.raises(TypeError):
        ipol.IpolBase(
            np.arange(12).reshape((2, 3, 2)), np.arange(20).reshape((2, 2, 5))
        )


@pytest.fixture(params=["linear", "nearest", "splinef2d"])
def get_rect_method(request):
    return request.param


@pytest.fixture
def rect_ipol():
    @dataclass(init=False, repr=False, eq=False)
    class TestRectGridInterpolation:
        ul = (2, 55)
        lr = (12, 45)

        x = np.linspace(ul[0], lr[0], 101)
        y = np.linspace(ul[1], lr[1], 51)

        grids = {}
        valgrids = {}
        X, Y = np.meshgrid(x, y)
        grids["image_upper"] = np.stack((X, Y), axis=-1)
        valgrids["image_upper"] = X + Y

        y = np.flip(y)

        X, Y = np.meshgrid(x, y)
        grids["image_lower"] = np.stack((X, Y), axis=-1)
        valgrids["image_lower"] = X + Y

        Y, X = np.meshgrid(y, x)
        grids["plot"] = np.stack((X, Y), axis=-1)
        valgrids["plot"] = X + Y

        xt = np.random.uniform(ul[0], lr[0], 10000)
        yt = np.random.uniform(lr[1], ul[1], 10000)
        points = np.stack((xt, yt), axis=-1)
        valpoints = xt + yt

        grid = grids["image_upper"]
        valgrid = valgrids["image_upper"]
        grid2 = (grid - ul) / 2 + ul
        valgrid2 = valgrid / 2

    yield TestRectGridInterpolation


def test_rect_grid(rect_ipol, get_rect_method):
    for indexing, src in rect_ipol.grids.items():
        ip = ipol.RectGrid(src, rect_ipol.points, method=get_rect_method)
        assert ("image" in indexing) == ip.image
        assert ("upper" in indexing) == ip.upper
        valip = ip(rect_ipol.valgrids[indexing])
        bad = np.isnan(valip)
        pbad = np.sum(bad) / bad.size
        assert pbad == 0

    for trg in rect_ipol.grids.values():
        ip = ipol.RectGrid(rect_ipol.grid2, trg, method=get_rect_method)
        valip = ip(rect_ipol.valgrid2)
        assert valip.shape == trg.shape[:-1]
        bad = np.isnan(valip)
        pbad = np.sum(bad) / bad.size
        assert abs(pbad - 0.75) < 0.1

        ip = ipol.RectGrid(rect_ipol.grid, rect_ipol.grid2, method=get_rect_method)
        valip = ip(rect_ipol.valgrid)
        bad = np.isnan(valip)
        pbad = np.sum(bad) / bad.size
        assert pbad == 0

        ip2 = ipol.RectGrid(rect_ipol.grid2, rect_ipol.grid, method=get_rect_method)
        valip2 = ip2(valip)
        bad = np.isnan(valip2)
        pbad = np.sum(bad) / bad.size
        assert abs(pbad - 0.75) < 0.1
        np.testing.assert_allclose(rect_ipol.valgrid[~bad], valip2[~bad])


def test_rect_bin(rect_ipol):
    ip = ipol.RectBin(rect_ipol.points, rect_ipol.grid)
    valip = ip(rect_ipol.valpoints)
    assert valip.shape == rect_ipol.grid.shape[:-1]

    grid2 = rect_ipol.grid2 + (-0.01, 0.01)
    ip = ipol.RectBin(grid2, rect_ipol.grid)
    valip = ip(rect_ipol.valgrid2)
    assert valip.shape == rect_ipol.grid.shape[:-1]

    bad = np.isnan(valip)
    pbad = np.sum(bad) / bad.size
    assert abs(pbad - 0.75) < 0.1

    firstcell = rect_ipol.valgrid2[0:2, 0:2]
    mean = np.mean(firstcell.ravel())
    np.testing.assert_allclose(mean, valip[0, 0])

    ip = ipol.RectBin(rect_ipol.points, rect_ipol.grid)
    ip(rect_ipol.valpoints, statistic="median")

    ip = ipol.RectBin(rect_ipol.points, rect_ipol.grid)
    res0 = ip(rect_ipol.valpoints, statistic="median")
    res0a = ip.binned_stats.statistic.copy()
    if ip.upper:
        res0a = np.flip(res0a, ip.ydim)
    res1 = ip(rect_ipol.valpoints, statistic="median")

    np.testing.assert_allclose(res0, res1)
    np.testing.assert_allclose(res0a, res1)


@requires_gdal
def test_QuadriArea(rect_ipol):
    grid2 = rect_ipol.grid2 + (-0.01, 0.01)
    ip = ipol.QuadriArea(grid2, rect_ipol.grid)
    valgrid2 = rect_ipol.valgrid2[1:, 1:]
    valip = ip(valgrid2)
    assert valip.shape == tuple([el - 1 for el in rect_ipol.grid.shape[:-1]])

    bad = np.isnan(valip)
    pbad = np.sum(bad) / bad.size
    assert abs(pbad - 0.75) < 0.1

    firstcell = valgrid2[0:3, 0:3]
    weights = np.array(
        [
            [72 / 100, 9 / 10, 18 / 100],
            [8 / 10, 1, 2 / 10],
            [8 / 100, 1 / 10, 2 / 100],
        ]
    )
    ref = np.sum(np.multiply(firstcell, weights)) / np.sum(weights)
    np.testing.assert_allclose(ref, valip[0, 0])


def test_IpolChain(rect_ipol):
    ip1 = ipol.RectGrid(rect_ipol.grid, rect_ipol.points)
    ip2 = ipol.RectGrid(rect_ipol.grid, rect_ipol.points)
    ipol.IpolChain((ip1, ip2))
    # ToDo: Needs more testing


def test_interpolate():
    src = np.arange(10)[:, None]
    trg = np.linspace(0, 20, 40)[:, None]
    vals = np.hstack((np.sin(src), 10.0 + np.sin(src)))
    vals[3:5, 1] = np.nan
    ipol_result = ipol.interpolate(
        src, trg, vals, ipol.Idw, remove_missing=True, nnearest=2
    )

    np.testing.assert_allclose(ipol_result[3:5, 1], np.array([10.880571, 10.909297]))

    ipol_result = ipol.interpolate(
        src, trg, vals[:, 1], ipol.Idw, remove_missing=True, nnearest=2
    )
    np.testing.assert_allclose(ipol_result[3:5], np.array([10.880571, 10.909136]))

    vals = np.dstack((np.sin(src), 10.0 + np.sin(src)))
    vals[3:5, :, 1] = np.nan
    with pytest.raises(NotImplementedError):
        ipol.interpolate(src, trg, vals, ipol.Idw, nnearest=2)


def test_interpolate_polar():
    data = np.arange(12.0).reshape(4, 3)
    masked_values = (data == 2) | (data == 9)
    filled_a = ipol.interpolate_polar(data, mask=masked_values, ipclass=ipol.Linear)
    testfunc = ipol.interpolate_polar
    with pytest.raises(ipol.MissingTargetsError):
        testfunc(data, mask=None, ipclass=ipol.Linear)
    mdata = np.ma.array(data, mask=masked_values)
    filled_b = ipol.interpolate_polar(mdata, ipclass=ipol.Linear)
    np.testing.assert_allclose(filled_a, filled_b)


@pytest.fixture
def reg_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestRegularToIrregular:
        NX = 2
        nx = np.linspace(-NX + 0.5, NX - 0.5, num=2 * NX, endpoint=True)
        vx = np.linspace(-NX, NX, num=2 * NX, endpoint=True)
        meshx, meshy = np.meshgrid(nx, nx)
        cartgrid = np.dstack((meshx, meshy))
        values = np.repeat(vx[:, np.newaxis], 2 * NX, 1)

        coord = georef.sweep_centroids(4, 1, NX, 0.0)
        xx = coord[..., 0]
        yy = coord[..., 1]

        xxx = xx * np.cos(np.radians(90.0 - yy))
        x = xx * np.sin(np.radians(90.0 - yy))
        y = xxx

        newgrid = np.dstack((x, y))

        result = np.array(
            [
                [0.47140452, 1.41421356],
                [0.47140452, 1.41421356],
                [-0.47140452, -1.41421356],
                [-0.47140452, -1.41421356],
            ]
        )

    yield TestRegularToIrregular


def test_cart_to_irregular_interp(reg_data):
    newvalues = ipol.cart_to_irregular_interp(
        reg_data.cartgrid, reg_data.values, reg_data.newgrid, method="linear"
    )
    assert np.allclose(newvalues, reg_data.result)


def test_cart_to_irregular_spline(reg_data):
    newvalues = ipol.cart_to_irregular_spline(
        reg_data.cartgrid, reg_data.values, reg_data.newgrid, order=1, prefilter=False
    )
    assert np.allclose(newvalues, reg_data.result)


def test_cart_to_irregular_equality(reg_data):
    assert np.allclose(
        ipol.cart_to_irregular_interp(
            reg_data.cartgrid, reg_data.values, reg_data.newgrid, method="linear"
        ),
        ipol.cart_to_irregular_spline(
            reg_data.cartgrid,
            reg_data.values,
            reg_data.newgrid,
            order=1,
            prefilter=False,
        ),
    )
