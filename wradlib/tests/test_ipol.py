#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.ipol as ipol
import wradlib.georef as georef
import unittest
import warnings


class InterpolationTest(unittest.TestCase):
    def setUp(self):
        # Kriging Variables
        self.src = np.array([[0., 0.], [4., 0]])
        self.trg = np.array([[0., 0.], [2., 0.], [1., 0], [4., 0]])
        self.src_d = np.array([0., 1.])
        self.trg_d = np.array([0., 1., 2., 3.])
        self.vals = np.array([[1., 2., 3.],
                              [3., 2., 1.]])

    def test_parse_covariogram(self):
        cov_model = '1.0 Exp(10.5) + 2.3 Sph(20.4) + 5.0 Nug(0.)'
        h = 5.0
        c = ipol.parse_covariogram(cov_model)
        ci = sum([ipol.cov_exp(h, 1., 10.5),
                  ipol.cov_sph(h, 2.3, 20.4),
                  ipol.cov_nug(h, 5.0, 0.)])
        self.assertTrue(c(h) == ci)

    def test_cov_lin(self):
        self.assertTrue(
            np.allclose(ipol.cov_lin([0., 5., 10.]), np.array([1., 0., 0.])))
        self.assertTrue(
            np.allclose(ipol.cov_lin([0., 5., 10.], sill=2., rng=10.),
                        np.array([2., 1., 0.])))

    def test_cov_sph(self):
        self.assertTrue(
            np.allclose(ipol.cov_sph([0., 5., 10.]), np.array([1., 0., 0.])))
        self.assertTrue(
            np.allclose(ipol.cov_sph([0., 5., 10.], sill=2., rng=10.),
                        np.array([2., 0.625, 0.])))

    def test_cov_exp(self):
        self.assertTrue(np.allclose(ipol.cov_exp([0., 5., 10.]), np.array(
            [1., 6.73794700e-03, 4.53999298e-05])))
        self.assertTrue(
            np.allclose(ipol.cov_exp([0., 5., 10.], sill=2., rng=10.),
                        np.array([2., 1.21306132, 0.73575888])))

    def test_cov_pow(self):
        self.assertTrue(
            np.allclose(ipol.cov_pow([0., 5., 10.]), np.array([1., -4., -9.])))
        self.assertTrue(
            np.allclose(ipol.cov_pow([0., 5., 10.], sill=2., rng=10.),
                        np.array([2.00000000e+00, -9.76562300e+06,
                                  -1.00000000e+10])))

    def test_cov_mat(self):
        self.assertTrue(np.allclose(ipol.cov_mat([0., 5., 10.]),
                                    np.array([1.00000000e+00, 8.49325705e-04,
                                              7.21354153e-07])))
        self.assertTrue(
            np.allclose(ipol.cov_mat([0., 5., 10.], sill=2., rng=10.),
                        np.array([2., 0.98613738, 0.48623347])))
        self.assertTrue(np.allclose(
            ipol.cov_mat([0., 5., 10.], sill=2., rng=10., shp=0.25),
            np.array([2., 0.74916629, 0.39961004])))

    def test_cov_gau(self):
        self.assertTrue(np.allclose(ipol.cov_gau([0., 5., 10.]),
                                    np.array([1.00000000e+00, 1.38879439e-11,
                                              3.72007598e-44])))
        self.assertTrue(
            np.allclose(ipol.cov_gau([0., 5., 10.], sill=2., rng=10.),
                        np.array([2., 1.55760157, 0.73575888])))

    def test_cov_cau(self):
        self.assertTrue(np.allclose(ipol.cov_cau([0., 5., 10.]),
                                    np.array([1., 0.16666667, 0.09090909])))
        self.assertTrue(
            np.allclose(ipol.cov_cau([0., 5., 10.], sill=2., rng=10., ),
                        np.array([2., 1.33333333, 1.])))
        self.assertTrue(np.allclose(
            ipol.cov_cau([0., 5., 10.], sill=2., rng=10., alpha=0.5),
            np.array([2., 0.6862915, 0.5])))
        self.assertTrue(np.allclose(
            ipol.cov_cau([0., 5., 10.], sill=2., rng=10., alpha=0.5, beta=1.5),
            np.array([2., 0.40202025, 0.25])))

    def test_OrdinaryKriging_1(self):
        """testing the basic behaviour of the OrdinaryKriging class"""

        ip = ipol.OrdinaryKriging(self.src, self.trg, '1.0 Lin(2.0)')

        res = ip(self.vals)
        self.assertTrue(np.all(res == np.array([[1., 2., 3.],
                                                [2., 2., 2.],
                                                [1.5, 2., 2.5],
                                                [3., 2., 1.]])))

    def test_ExternalDriftKriging_1(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with drift terms constant over multiple fields"""

        ip = ipol.ExternalDriftKriging(self.src, self.trg, '1.0 Lin(2.0)',
                                       src_drift=self.src_d,
                                       trg_drift=self.trg_d)

        res = ip(self.vals)
        self.assertTrue(np.all(res == np.array([[1., 2., 3.],
                                                [3., 2., 1.],
                                                [5., 2., -1.],
                                                [7., 2., -3.]])))

    def test_ExternalDriftKriging_2(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with drift terms varying over multiple fields"""
        src_d = np.array([[0., 0., 0.],
                          [1., 1., 1.]])
        trg_d = np.array([[0., 0., 0.],
                          [1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.]])

        ip = ipol.ExternalDriftKriging(self.src, self.trg, '1.0 Lin(2.0)',
                                       src_drift=src_d,
                                       trg_drift=trg_d)

        res = ip(self.vals)
        self.assertTrue(np.all(res == np.array([[1., 2., 3.],
                                                [3., 2., 1.],
                                                [5., 2., -1.],
                                                [7., 2., -3.]])))

    def test_ExternalDriftKriging_3(self):
        """testing the basic behaviour of the ExternalDriftKriging class
        with missing drift terms"""
        ip = ipol.ExternalDriftKriging(self.src, self.trg, '1.0 Lin(2.0)',
                                       src_drift=None,
                                       trg_drift=None)

        self.assertRaises(ValueError, ip, self.vals)

    def test_MissingErrors(self):
        self.assertRaises(ipol.MissingSourcesError,
                          ipol.Nearest, np.array([]), self.trg)
        self.assertRaises(ipol.MissingTargetsError,
                          ipol.Nearest, self.src, np.array([]))
        self.assertRaises(ipol.MissingSourcesError,
                          ipol.Idw, np.array([]), self.trg)
        self.assertRaises(ipol.MissingTargetsError,
                          ipol.Idw, self.src, np.array([]))
        self.assertRaises(ipol.MissingSourcesError,
                          ipol.Linear, np.array([]), self.trg)
        self.assertRaises(ipol.MissingTargetsError,
                          ipol.Linear, self.src, np.array([]))
        self.assertRaises(ipol.MissingSourcesError,
                          ipol.OrdinaryKriging, np.array([]), self.trg)
        self.assertRaises(ipol.MissingTargetsError,
                          ipol.OrdinaryKriging, self.src, np.array([]))
        self.assertRaises(ipol.MissingSourcesError,
                          ipol.ExternalDriftKriging, np.array([]), self.trg)
        self.assertRaises(ipol.MissingTargetsError,
                          ipol.ExternalDriftKriging, self.src, np.array([]))

    def test_nnearest_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ipol.Idw(self.src, self.trg, nnearest=len(self.src) + 1)
            ipol.OrdinaryKriging(self.src, self.trg,
                                 nnearest=len(self.src) + 1)
            ipol.ExternalDriftKriging(self.src, self.trg,
                                      nnearest=len(self.src) + 1)
            for item in w:
                self.assertTrue(issubclass(item.category, UserWarning))
                self.assertTrue("nnearest" in str(item.message))

    def test_IpolBase(self):
        """testing the basic behaviour of the base class"""

        ip = ipol.IpolBase(self.src, self.trg)
        res = ip(self.vals)
        self.assertEqual(res, None)

        # Check behaviour if args are passed as lists
        src = [self.src[:, 0], self.src[:, 1]]
        trg = [self.trg[:, 0], self.trg[:, 1]]
        ip = ipol.IpolBase(src, trg)
        self.assertEqual(len(self.src), ip.numsources)

        # Check behaviour if dimension is > 2
        ip = ipol.IpolBase(self.src, self.trg)
        self.assertRaises(Exception, ipol.IpolBase,
                          np.arange(12).reshape((2, 3, 2)),
                          np.arange(20).reshape((2, 2, 5)))


class Regular2IrregularTest(unittest.TestCase):
    def setUp(self):
        NX = 2
        nx = np.linspace(-NX + 0.5, NX - 0.5, num=2 * NX, endpoint=True)
        vx = np.linspace(-NX, NX, num=2 * NX, endpoint=True)
        meshx, meshy = np.meshgrid(nx, nx)
        self.cartgrid = np.dstack((meshx, meshy))
        self.values = np.repeat(vx[:, np.newaxis], 2 * NX, 1)

        coord = georef.sweep_centroids(4, 1, NX, 0.)
        xx = coord[..., 0]
        yy = np.degrees(coord[..., 1])

        xxx = xx * np.cos(np.radians(90. - yy))
        x = xx * np.sin(np.radians(90. - yy))
        y = xxx

        self.newgrid = np.dstack((x, y))

        self.result = np.array([[0.47140452, 1.41421356],
                                [0.47140452, 1.41421356],
                                [-0.47140452, -1.41421356],
                                [-0.47140452, -1.41421356]])

    def test_cart2irregular_interp(self):
        newvalues = ipol.cart2irregular_interp(self.cartgrid, self.values,
                                               self.newgrid, method='linear')
        print(newvalues)
        self.assertTrue(np.allclose(newvalues, self.result))

    def test_cart2irregular_spline(self):
        newvalues = ipol.cart2irregular_spline(self.cartgrid, self.values,
                                               self.newgrid, order=1,
                                               prefilter=False)
        print(newvalues)
        self.assertTrue(np.allclose(newvalues, self.result))

    def test_cart2irregular_equality(self):
        self.assertTrue(
            np.allclose(ipol.cart2irregular_interp(self.cartgrid, self.values,
                                                   self.newgrid,
                                                   method='linear'),
                        ipol.cart2irregular_spline(self.cartgrid, self.values,
                                                   self.newgrid,
                                                   order=1, prefilter=False)))


if __name__ == '__main__':
    unittest.main()
