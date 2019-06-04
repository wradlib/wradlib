#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import numpy as np
import matplotlib.pyplot as pl
pl.interactive(True)  # noqa
from .. import verify
from .. import georef


class PolarNeighboursTest(unittest.TestCase):
    def setUp(self):
        self.r = np.arange(1, 100, 10)
        self.az = np.arange(0, 360, 90)
        self.site = (9.7839, 48.5861)
        self.proj = georef.epsg_to_osr(31467)
        # Coordinates of the rain gages in Gauss-Krueger 3 coordinates
        self.x, self.y = (np.array([3557880, 3557890]),
                          np.array([5383379, 5383375]))

        np.random.seed(42)
        self.data = np.random.random((len(self.az), len(self.r)))

    def test___init__(self):
        verify.PolarNeighbours(self.r, self.az, self.site, self.proj,
                               self.x, self.y, nnear=9)

    def test_extract(self):
        pn = verify.PolarNeighbours(self.r, self.az, self.site, self.proj,
                                    self.x, self.y, nnear=4)
        neighbours = pn.extract(self.data)
        res0 = np.array([0.59241457, 0.04645041, 0.51423444, 0.19967378])
        res1 = np.array([0.04645041, 0.59241457, 0.51423444, 0.19967378])
        np.testing.assert_allclose(neighbours[0], res0)
        np.testing.assert_allclose(neighbours[1], res1)

    def test_get_bincoords(self):
        pn = verify.PolarNeighbours(self.r, self.az, self.site, self.proj,
                                    self.x, self.y, nnear=4)
        bx, by = pn.get_bincoords()
        np.testing.assert_almost_equal(bx[0], 3557908.88665658)
        np.testing.assert_almost_equal(by[0], 5383452.639404042)

    def test_get_bincoords_at_points(self):
        pn = verify.PolarNeighbours(self.r, self.az, self.site, self.proj,
                                    self.x, self.y, nnear=4)
        bx, by = pn.get_bincoords_at_points()
        resx0 = np.array([3557909.62605379, 3557909.72874732, 3557909.52336013,
                          3557909.42066632])
        resx1 = np.array([3557909.72874732, 3557909.62605379, 3557909.52336013,
                          3557909.42066632])

        resy0 = np.array([5383380.64013055, 5383370.64023136, 5383390.64002972,
                          5383400.6399289])
        resy1 = np.array([5383370.64023136, 5383380.64013055, 5383390.64002972,
                          5383400.6399289])

        np.testing.assert_allclose(bx[0], resx0)
        np.testing.assert_allclose(bx[1], resx1)
        np.testing.assert_allclose(by[0], resy0)
        np.testing.assert_allclose(by[1], resy1)


class ErrorMetricsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.obs = np.random.uniform(0, 10, 100)
        self.est = np.random.uniform(0, 10, 100)
        self.non = np.zeros(100) * np.nan

    def test___init__(self):
        self.metrics = verify.ErrorMetrics(self.obs, self.est)
        with self.assertRaises(ValueError):
            verify.ErrorMetrics(self.obs, self.est[:10])

    def test___init__warn(self):
        with self.assertWarns(UserWarning):
            verify.ErrorMetrics(self.obs, self.non)

    def test_all_metrics(self):
        metrics = verify.ErrorMetrics(self.obs, self.est)
        metrics.all()

    def test_pprint(self):
        metrics = verify.ErrorMetrics(self.obs, self.est)
        metrics.pprint()


if __name__ == '__main__':
    unittest.main()
