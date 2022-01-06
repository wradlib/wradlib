#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import warnings

import numpy as np
import pytest

from wradlib import georef, ipol

from . import requires_gdal


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

    def test_parse_covariogram(self):
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

    def test_cov_lin(self):
        assert np.allclose(ipol.cov_lin([0.0, 5.0, 10.0]), np.array([1.0, 0.0, 0.0]))
        assert np.allclose(
            ipol.cov_lin([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
            np.array([2.0, 1.0, 0.0]),
        )

    def test_cov_sph(self):
        assert np.allclose(ipol.cov_sph([0.0, 5.0, 10.0]), np.array([1.0, 0.0, 0.0]))
        assert np.allclose(
            ipol.cov_sph([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
            np.array([2.0, 0.625, 0.0]),
        )

    def test_cov_exp(self):
        assert np.allclose(
            ipol.cov_exp([0.0, 5.0, 10.0]),
            np.array([1.0, 6.73794700e-03, 4.53999298e-05]),
        )
        assert np.allclose(
            ipol.cov_exp([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
            np.array([2.0, 1.21306132, 0.73575888]),
        )

    def test_cov_pow(self):
        assert np.allclose(ipol.cov_pow([0.0, 5.0, 10.0]), np.array([1.0, -4.0, -9.0]))
        assert np.allclose(
            ipol.cov_pow([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
            np.array([2.00000000e00, -9.76562300e06, -1.00000000e10]),
        )

    def test_cov_mat(self):
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

    def test_cov_gau(self):
        assert np.allclose(
            ipol.cov_gau([0.0, 5.0, 10.0]),
            np.array([1.00000000e00, 1.38879439e-11, 3.72007598e-44]),
        )
        assert np.allclose(
            ipol.cov_gau([0.0, 5.0, 10.0], sill=2.0, rng=10.0),
            np.array([2.0, 1.55760157, 0.73575888]),
        )

    def test_cov_cau(self):
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

    def test_Nearest_1(self):
        """testing the basic behaviour of the Idw class"""
        ip = ipol.Nearest(self.src, self.trg)
        # input more than one dataset
        res = ip(self.vals)
        assert np.allclose(
            res,
            np.array(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
            ),
        )
        # input only one flat array
        res = ip(self.vals[:, 2])
        assert np.allclose(res, np.array([3.0, 3.0, 3.0, 1.0]))

    def test_Idw_1(self):
        """testing the basic behaviour of the Idw class"""
        ip = ipol.Idw(self.src, self.trg)
        # input more than one dataset
        res = ip(self.vals)
        assert np.allclose(
            res,
            np.array(
                [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.2, 2.0, 2.8], [3.0, 2.0, 1.0]]
            ),
        )
        # input only one flat array
        res = ip(self.vals[:, 2])
        assert np.allclose(res, np.array([3.0, 2.0, 2.8, 1.0]))

    def test_Linear_1(self):
        """testing the basic behaviour of the Linear class"""
        ip = ipol.Linear(self.src_lin, self.trg_lin)
        # input more than one dataset
        res = ip(self.vals_lin)
        assert np.allclose(
            res,
            np.array(
                [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.5, 2.0, 2.5], [3.0, 2.0, 1.0]]
            ),
        )
        # input only one flat array
        res = ip(self.vals_lin[:, 2])
        assert np.allclose(res, np.array([3.0, 2.0, 2.5, 1.0]))

    def test_OrdinaryKriging_1(self):
        """testing the basic behaviour of the OrdinaryKriging class"""

        ip = ipol.OrdinaryKriging(self.src, self.trg, "1.0 Lin(2.0)")
        # input more than one dataset
        res = ip(self.vals)
        assert np.all(
            res
            == np.array(
                [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.5, 2.0, 2.5], [3.0, 2.0, 1.0]]
            )
        )
        # input only one flat array
        res = ip(self.vals[:, 2])
        assert np.allclose(res, np.array([3.0, 2.0, 2.5, 1.0]))

    def test_ExternalDriftKriging_1(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with drift terms constant over multiple fields"""

        ip = ipol.ExternalDriftKriging(
            self.src,
            self.trg,
            "1.0 Lin(2.0)",
            src_drift=self.src_d,
            trg_drift=self.trg_d,
        )

        # input more than one dataset
        res = ip(self.vals)
        assert np.all(
            res
            == np.array(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 2.0, -1.0], [7.0, 2.0, -3.0]]
            )
        )
        # input only one flat array
        res = ip(self.vals[:, 2])
        assert np.allclose(res, np.array([3.0, 1.0, -1.0, -3.0]))

    def test_ExternalDriftKriging_2(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with drift terms varying over multiple fields"""
        src_d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        trg_d = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        )

        ip = ipol.ExternalDriftKriging(
            self.src, self.trg, "1.0 Lin(2.0)", src_drift=src_d, trg_drift=trg_d
        )

        res = ip(self.vals)
        assert np.all(
            res
            == np.array(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 2.0, -1.0], [7.0, 2.0, -3.0]]
            )
        )
        # input only one flat array
        res = ip(self.vals[:, 2], src_drift=src_d[:, 2], trg_drift=trg_d[:, 2])
        assert np.allclose(res, np.array([3.0, 1.0, -1.0, -3.0]))

    def test_ExternalDriftKriging_3(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with missing drift terms"""
        ip = ipol.ExternalDriftKriging(
            self.src, self.trg, "1.0 Lin(2.0)", src_drift=None, trg_drift=None
        )

        with pytest.raises(ValueError):
            ip(self.vals)

    def test_MissingErrors(self):
        with pytest.raises(ipol.MissingSourcesError):
            ipol.Nearest(np.array([]), self.trg)
        with pytest.raises(ipol.MissingTargetsError):
            ipol.Nearest(self.src, np.array([]))
        with pytest.raises(ipol.MissingSourcesError):
            ipol.Idw(np.array([]), self.trg)
        with pytest.raises(ipol.MissingTargetsError):
            ipol.Idw(self.src, np.array([]))
        with pytest.raises(ipol.MissingSourcesError):
            ipol.Linear(np.array([]), self.trg)
        with pytest.raises(ipol.MissingTargetsError):
            ipol.Linear(self.src, np.array([]))
        with pytest.raises(ipol.MissingSourcesError):
            ipol.OrdinaryKriging(np.array([]), self.trg)
        with pytest.raises(ipol.MissingTargetsError):
            ipol.OrdinaryKriging(self.src, np.array([]))
        with pytest.raises(ipol.MissingSourcesError):
            ipol.ExternalDriftKriging(np.array([]), self.trg)
        with pytest.raises(ipol.MissingTargetsError):
            ipol.ExternalDriftKriging(self.src, np.array([]))

    def test_nnearest_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ipol.Idw(self.src, self.trg, nnearest=len(self.src) + 1)
            ipol.OrdinaryKriging(self.src, self.trg, nnearest=len(self.src) + 1)
            ipol.ExternalDriftKriging(self.src, self.trg, nnearest=len(self.src) + 1)
            for item in w:
                assert issubclass(item.category, UserWarning)
                assert "nnearest" in str(item.message)

    def test_IpolBase(self):
        """testing the basic behaviour of the base class"""

        ip = ipol.IpolBase(self.src, self.trg)
        res = ip(self.vals)
        assert res is None

        # Check behaviour if args are passed as lists
        src = [self.src[:, 0], self.src[:, 1]]
        trg = [self.trg[:, 0], self.trg[:, 1]]
        ip = ipol.IpolBase(src, trg)
        assert len(self.src) == ip.numsources

        # Check behaviour if dimension is > 2
        ip = ipol.IpolBase(self.src, self.trg)
        with pytest.raises(Exception):
            ipol.IpolBase(
                np.arange(12).reshape((2, 3, 2)), np.arange(20).reshape((2, 2, 5))
            )


@pytest.fixture(params=["linear", "nearest", "splinef2d"])
def get_rect_method(request):
    return request.param


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

    def test_rect_grid(self, get_rect_method):
        for indexing, src in self.grids.items():
            ip = ipol.RectGrid(src, self.points, method=get_rect_method)
            assert ("image" in indexing) == ip.image
            assert ("upper" in indexing) == ip.upper
            valip = ip(self.valgrids[indexing])
            bad = np.isnan(valip)
            pbad = np.sum(bad) / bad.size
            assert pbad == 0

        for indexing, trg in self.grids.items():
            ip = ipol.RectGrid(self.grid2, trg, method=get_rect_method)
            valip = ip(self.valgrid2)
            assert valip.shape == trg.shape[:-1]
            bad = np.isnan(valip)
            pbad = np.sum(bad) / bad.size
            assert abs(pbad - 0.75) < 0.1

            ip = ipol.RectGrid(self.grid, self.grid2, method=get_rect_method)
            valip = ip(self.valgrid)
            bad = np.isnan(valip)
            pbad = np.sum(bad) / bad.size
            assert pbad == 0

            ip2 = ipol.RectGrid(self.grid2, self.grid, method=get_rect_method)
            valip2 = ip2(valip)
            bad = np.isnan(valip2)
            pbad = np.sum(bad) / bad.size
            assert abs(pbad - 0.75) < 0.1
            np.testing.assert_allclose(self.valgrid[~bad], valip2[~bad])

    def test_rect_bin(self):
        ip = ipol.RectBin(self.points, self.grid)
        valip = ip(self.valpoints)
        assert valip.shape == self.grid.shape[:-1]

        grid2 = self.grid2 + (-0.01, 0.01)
        ip = ipol.RectBin(grid2, self.grid)
        valip = ip(self.valgrid2)
        assert valip.shape == self.grid.shape[:-1]

        bad = np.isnan(valip)
        pbad = np.sum(bad) / bad.size
        assert abs(pbad - 0.75) < 0.1

        firstcell = self.valgrid2[0:2, 0:2]
        mean = np.mean(firstcell.ravel())
        np.testing.assert_allclose(mean, valip[0, 0])

        ip = ipol.RectBin(self.points, self.grid)
        ip(self.valpoints, statistic="median")

        ip = ipol.RectBin(self.points, self.grid)
        res0 = ip(self.valpoints, statistic="median")
        res0a = ip.binned_stats.statistic.copy()
        if ip.upper:
            res0a = np.flip(res0a, ip.ydim)
        res1 = ip(self.valpoints, statistic="median")

        np.testing.assert_allclose(res0, res1)
        np.testing.assert_allclose(res0a, res1)

    @requires_gdal
    def test_QuadriArea(self):
        grid2 = self.grid2 + (-0.01, 0.01)
        ip = ipol.QuadriArea(grid2, self.grid)
        valgrid2 = self.valgrid2[1:, 1:]
        valip = ip(valgrid2)
        assert valip.shape == tuple([el - 1 for el in self.grid.shape[:-1]])

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

    def test_IpolChain(self):
        ip1 = ipol.RectGrid(self.grid, self.points)
        ip2 = ipol.RectGrid(self.grid, self.points)
        ipol.IpolChain((ip1, ip2))
        # ToDo: Needs more testing


class TestWrapperFunction:
    def test_interpolate(self):
        src = np.arange(10)[:, None]
        trg = np.linspace(0, 20, 40)[:, None]
        vals = np.hstack((np.sin(src), 10.0 + np.sin(src)))
        vals[3:5, 1] = np.nan
        ipol_result = ipol.interpolate(
            src, trg, vals, ipol.Idw, remove_missing=True, nnearest=2
        )

        np.testing.assert_allclose(
            ipol_result[3:5, 1], np.array([10.880571, 10.909297])
        )

        ipol_result = ipol.interpolate(
            src, trg, vals[:, 1], ipol.Idw, remove_missing=True, nnearest=2
        )
        np.testing.assert_allclose(ipol_result[3:5], np.array([10.880571, 10.909136]))

        vals = np.dstack((np.sin(src), 10.0 + np.sin(src)))
        vals[3:5, :, 1] = np.nan
        with pytest.raises(NotImplementedError):
            ipol.interpolate(src, trg, vals, ipol.Idw, nnearest=2)

    def test_interpolate_polar(self):
        data = np.arange(12.0).reshape(4, 3)
        masked_values = (data == 2) | (data == 9)
        filled_a = ipol.interpolate_polar(data, mask=masked_values, ipclass=ipol.Linear)
        testfunc = ipol.interpolate_polar
        with pytest.raises(ipol.MissingTargetsError):
            testfunc(data, mask=None, ipclass=ipol.Linear)
        mdata = np.ma.array(data, mask=masked_values)
        filled_b = ipol.interpolate_polar(mdata, ipclass=ipol.Linear)
        np.testing.assert_allclose(filled_a, filled_b)


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

    def test_cart_to_irregular_interp(self):
        newvalues = ipol.cart_to_irregular_interp(
            self.cartgrid, self.values, self.newgrid, method="linear"
        )
        assert np.allclose(newvalues, self.result)

    def test_cart_to_irregular_spline(self):
        newvalues = ipol.cart_to_irregular_spline(
            self.cartgrid, self.values, self.newgrid, order=1, prefilter=False
        )
        assert np.allclose(newvalues, self.result)

    def test_cart_to_irregular_equality(self):
        assert np.allclose(
            ipol.cart_to_irregular_interp(
                self.cartgrid, self.values, self.newgrid, method="linear"
            ),
            ipol.cart_to_irregular_spline(
                self.cartgrid, self.values, self.newgrid, order=1, prefilter=False
            ),
        )
