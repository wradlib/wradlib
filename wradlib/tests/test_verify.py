#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import pytest

from wradlib import georef, verify


class TestPolarNeighbours:
    r = np.arange(1, 100, 10)
    az = np.arange(0, 360, 90)
    site = (9.7839, 48.5861)
    proj = georef.epsg_to_osr(31467)
    # Coordinates of the rain gages in Gauss-Krueger 3 coordinates
    x, y = (np.array([3557880, 3557890]), np.array([5383379, 5383375]))

    np.random.seed(42)
    data = np.random.random((len(az), len(r)))

    def test___init__(self):
        verify.PolarNeighbours(
            self.r, self.az, self.site, self.proj, self.x, self.y, nnear=9
        )

    def test_extract(self):
        pn = verify.PolarNeighbours(
            self.r, self.az, self.site, self.proj, self.x, self.y, nnear=4
        )
        neighbours = pn.extract(self.data)
        res0 = np.array([0.59241457, 0.04645041, 0.51423444, 0.19967378])
        res1 = np.array([0.04645041, 0.59241457, 0.51423444, 0.19967378])
        np.testing.assert_allclose(neighbours[0], res0)
        np.testing.assert_allclose(neighbours[1], res1)

    def test_get_bincoords(self):
        pn = verify.PolarNeighbours(
            self.r, self.az, self.site, self.proj, self.x, self.y, nnear=4
        )
        bx, by = pn.get_bincoords()
        np.testing.assert_allclose(bx[0], 3557908.88665658, rtol=1e-6)
        np.testing.assert_allclose(by[0], 5383452.639404042, rtol=1e-6)

    def test_get_bincoords_at_points(self):
        pn = verify.PolarNeighbours(
            self.r, self.az, self.site, self.proj, self.x, self.y, nnear=4
        )
        bx, by = pn.get_bincoords_at_points()
        resx0 = np.array(
            [3557909.62605379, 3557909.72874732, 3557909.52336013, 3557909.42066632]
        )
        resx1 = np.array(
            [3557909.72874732, 3557909.62605379, 3557909.52336013, 3557909.42066632]
        )

        resy0 = np.array(
            [5383380.64013055, 5383370.64023136, 5383390.64002972, 5383400.6399289]
        )
        resy1 = np.array(
            [5383370.64023136, 5383380.64013055, 5383390.64002972, 5383400.6399289]
        )

        np.testing.assert_allclose(bx[0], resx0, rtol=1e-6)
        np.testing.assert_allclose(bx[1], resx1, rtol=1e-6)
        np.testing.assert_allclose(by[0], resy0, rtol=1e-6)
        np.testing.assert_allclose(by[1], resy1, rtol=1e-6)


class TestErrorMetrics:
    np.random.seed(42)
    obs = np.random.uniform(0, 10, 100)
    est = np.random.uniform(0, 10, 100)
    non = np.zeros(100) * np.nan

    def test___init__(self):
        self.metrics = verify.ErrorMetrics(self.obs, self.est)
        with pytest.raises(ValueError):
            verify.ErrorMetrics(self.obs, self.est[:10])

    def test___init__warn(self):
        with pytest.warns(UserWarning):
            verify.ErrorMetrics(self.obs, self.non)

    def test_all_metrics(self):
        metrics = verify.ErrorMetrics(self.obs, self.est)
        metrics.all()

    def test_pprint(self):
        metrics = verify.ErrorMetrics(self.obs, self.est)
        metrics.pprint()
