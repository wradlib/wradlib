#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import sys
import pytest

import numpy as np

from wradlib import dp


@pytest.fixture(params=['lstsq', 'cov', 'matrix_inv',
                        'lanczos_conv', 'lanczos_dot'])
def derivation_method(request):
    return request.param


class TestKDPFromPHIDP:
    np.random.seed(42)
    # Synthetic truth
    dr = 0.5
    r = np.arange(0, 100, dr)
    kdp_true0 = np.sin(0.3 * r)
    kdp_true0[kdp_true0 < 0] = 0.
    phidp_true0 = np.cumsum(kdp_true0) * 2 * dr
    # Synthetic observation of PhiDP with a random noise and gaps
    phidp_raw0 = (phidp_true0 + np.random.uniform(-2, 2, len(phidp_true0)))
    gaps = np.random.uniform(0, len(r), 20).astype("int")
    phidp_raw0[gaps] = np.nan
    rho = np.random.uniform(0.8, 1.0, len(r))

    # for derivation tests
    window = 7
    az = 360
    rng = 1000
    pad = window // 2
    kdp_true = np.arange(az * rng, dtype=np.float).reshape(az, rng)
    phidp_true = np.power(kdp_true, 2)
    dr = 0.1
    kdp_true /= (dr)
    phidp_true_nan = phidp_true.copy()
    phidp_true_nan[:, window:-1:10] = np.nan

    def test_process_raw_phidp_vulpiani(self):
        dp.process_raw_phidp_vulpiani(self.phidp_raw0, dr=self.dr,
                                      copy=True)
        dp.process_raw_phidp_vulpiani(self.phidp_raw0, dr=self.dr)

    def test_kdp_from_phidp(self, derivation_method):
        if (derivation_method == 'lstsq' and sys.platform.startswith("win")):
            pytest.skip("fails on windows due to MKL issue")

        # compare with true kdp
        out = dp.kdp_from_phidp(self.phidp_true, dr=self.dr,
                                method=derivation_method)
        outx = out[:, self.pad:-self.pad]
        res = self.kdp_true[:, self.pad:-self.pad]
        np.testing.assert_array_almost_equal(outx, res, decimal=4)

        # intercompare with lanczos method with NaN handling
        out0 = dp.kdp_from_phidp(self.phidp_true, dr=self.dr,
                                 method='lanczos_conv')
        np.testing.assert_array_almost_equal(out, out0, decimal=4)

        # intercompare with lanczos method without NaN-handling
        out0 = dp.kdp_from_phidp(self.phidp_true_nan, dr=self.dr,
                                 method='lanczos_conv', skipna=False)
        outx = dp.kdp_from_phidp(self.phidp_true_nan, dr=self.dr,
                                 method=derivation_method, skipna=False)
        np.testing.assert_array_almost_equal(outx, out0, decimal=4)

    def test_linear_despeckle(self):
        dp.linear_despeckle(self.phidp_raw0, ndespeckle=3, copy=True)
        dp.linear_despeckle(self.phidp_raw0, ndespeckle=5, copy=True)

    def test_unfold_phi_naive(self):
        dp.unfold_phi_naive(self.phidp_raw0, self.rho)
        dp.unfold_phi_naive(self.phidp_raw0, self.rho, copy=True)

    def test_unfold_phi_vulpiani(self):
        phi_true = np.arange(600)
        phi_raw1 = phi_true.copy()
        phi_raw1[phi_raw1 > 540] -= 360
        phi_raw2 = phi_raw1.copy()
        phi_raw2[phi_raw2 > 180] -= 360
        kdp1 = dp.kdp_from_phidp(phi_raw1)
        kdp2 = dp.kdp_from_phidp(phi_raw2)

        out1 = dp.unfold_phi_vulpiani(phi_raw1.copy(), kdp1)
        out2 = dp.unfold_phi_vulpiani(phi_raw2.copy(), kdp2)
        kdp3 = dp.kdp_from_phidp(out2)
        out3 = dp.unfold_phi_vulpiani(out2.copy(), kdp3)

        np.testing.assert_array_equal(out1, phi_true)
        np.testing.assert_array_equal(out2, phi_raw1)
        np.testing.assert_array_equal(out3, phi_true)

    def test__fill_sweep(self):
        dp._fill_sweep(self.phidp_raw0, kind='linear')


class TestTexture:
    img = np.zeros((360, 10), dtype=np.float32)
    img[2, 2] = 10  # isolated pixel
    img[5, 6:8] = 10  # line
    img[20, :] = 5  # spike
    img[60:120, 2:7] = 11  # precip field

    pixel = np.ones((3, 3)) * 3.5355339059327378
    pixel[1, 1] = 10.

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

    def test_texture(self):
        tex = dp.texture(self.img)
        np.testing.assert_array_equal(tex[1:4, 1:4], self.pixel)
        np.testing.assert_array_equal(tex[4:7, 5:9], self.line)
        np.testing.assert_array_equal(tex[19:22], self.spike)
        np.testing.assert_array_equal(tex[59:121, 1:8], self.rainfield)


class TestDepolarization:
    def test_depolarization(self):
        zdr = np.linspace(-0.5, 0.5, 10)
        rho = np.linspace(0., 1., 10)

        dr_0 = [-12.719937, -12.746507, -12.766551, -12.779969, -12.786695,
                -12.786695, -12.779969, -12.766551, -12.746507, -12.719937]
        dr_1 = [0., -0.96266, -1.949568, -2.988849, -4.118078, -5.394812,
                -6.921361, -8.919312, -12.067837, -24.806473]

        np.testing.assert_array_almost_equal(dp.depolarization(zdr, 0.9),
                                             dr_0)
        np.testing.assert_array_almost_equal(dp.depolarization(1.0, rho),
                                             dr_1)
