#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import sys

import numpy as np
import pytest
from scipy import integrate

from wradlib import dp


@pytest.fixture(params=["lstsq", "cov", "matrix_inv", "lanczos_conv", "lanczos_dot"])
def derivation_method(request):
    return request.param


@pytest.fixture(params=[11, 13, 15])
def window(request):
    return request.param


@pytest.fixture(params=[True, False])
def copy(request):
    return request.param


@pytest.fixture(params=[3, 5])
def ndespeckle(request):
    return request.param


class TestKDPFromPHIDP:
    np.random.seed(42)
    # Synthetic truth
    dr = 0.5
    r = np.arange(0, 100, dr)
    kdp_true0 = np.sin(0.3 * r)
    kdp_true0[kdp_true0 < 0] = 0.0
    phidp_true0 = np.cumsum(kdp_true0) * 2 * dr
    # Synthetic observation of PhiDP with a random noise and gaps
    phidp_raw0 = phidp_true0 + np.random.uniform(-2, 2, len(phidp_true0))
    gaps = np.random.uniform(0, len(r), 20).astype("int")
    phidp_raw0[gaps] = np.nan
    rho = np.random.uniform(0.8, 1.0, len(r))

    # for derivation tests
    window = 7
    az = 360
    rng = 1000
    pad = window // 2
    kdp_true = np.arange(az * rng, dtype=np.float_).reshape(az, rng)
    phidp_true = np.power(kdp_true, 2)
    dr = 0.1
    kdp_true /= dr
    phidp_true_nan = phidp_true.copy()
    phidp_true_nan[:, window:-1:10] = np.nan

    def test_process_raw_phidp_vulpiani(self, derivation_method, window, copy):
        if derivation_method == "lstsq" and sys.platform.startswith("win"):
            pytest.skip("fails on windows due to MKL issue")
        # Todo: move data setup into fixture
        np.random.seed(42000)
        # Synthetic truth
        dr = 0.5
        r = np.arange(0, 500, dr)

        kdp_true0 = np.sin(0.3 * r)
        kdp_true0[kdp_true0 < 0] = 0.0
        phidp_true0 = 2 * integrate.cumtrapz(kdp_true0, axis=-1, initial=0, dx=dr)
        fillval = phidp_true0[200]
        phidp_true0 = np.concatenate(
            (phidp_true0[:200], np.ones(20) * fillval, phidp_true0[200:])
        )
        phidp_true0 = np.stack([phidp_true0, phidp_true0], axis=0)

        # first, no noise, no folding, no gaps, offset
        phidp_raw0 = phidp_true0.copy() + 30.0

        # second, noise, no folding, no gaps
        phidp_raw1 = phidp_raw0.copy()
        phidp_raw1 += np.random.uniform(-2, 2, phidp_raw1.shape[-1])

        # third, noise, folding, no gaps
        phidp_raw2 = phidp_raw1.copy()
        phidp_raw2[phidp_raw2 > 180] -= 360

        # fourth, noise, folding, large gap
        phidp_raw3 = phidp_raw2.copy()
        phidp_raw3[:, 200:220] = np.nan

        # fifth, noise, folding, large gap, small gaps
        phidp_raw4 = phidp_raw3.copy()
        gaps = np.random.uniform(0, phidp_raw4.shape[-1], 50).astype("int")
        phidp_raw4[:, gaps] = np.nan

        in0 = phidp_raw0.copy()
        out0 = dp.process_raw_phidp_vulpiani(
            in0,
            dr=dr,
            copy=copy,
            winlen=window,
            method=derivation_method,
            pad_mode="reflect",
            pad_kwargs={"reflect_type": "odd"},
            niter=1,
        )
        np.testing.assert_array_equal(in0, phidp_raw0)
        np.testing.assert_allclose(out0[0], phidp_true0, atol=0.6, rtol=0.02)

        out1 = dp.process_raw_phidp_vulpiani(
            phidp_raw1.copy(),
            dr=dr,
            copy=copy,
            winlen=window,
            method=derivation_method,
            pad_mode="reflect",
            pad_kwargs={"reflect_type": "even"},
            niter=1,
        )
        np.testing.assert_allclose(out1[0], phidp_true0, atol=0.8, rtol=0.02)

        out2 = dp.process_raw_phidp_vulpiani(
            phidp_raw1.copy(),
            dr=dr,
            copy=copy,
            winlen=window,
            method=derivation_method,
            pad_mode="reflect",
            pad_kwargs={"reflect_type": "even"},
            niter=1,
        )
        np.testing.assert_allclose(out2[0], phidp_true0, atol=0.8, rtol=0.02)

        out3 = dp.process_raw_phidp_vulpiani(
            phidp_raw1.copy(),
            dr=dr,
            copy=copy,
            winlen=window,
            method=derivation_method,
            pad_mode="reflect",
            pad_kwargs={"reflect_type": "even"},
            niter=1,
        )
        np.testing.assert_allclose(out3[0], phidp_true0, atol=0.8, rtol=0.02)

        in4 = phidp_raw4.copy()
        out4 = dp.process_raw_phidp_vulpiani(
            in4,
            dr=dr,
            copy=copy,
            winlen=window,
            method=derivation_method,
            pad_mode="reflect",
            pad_kwargs={"reflect_type": "even"},
            niter=1,
        )
        np.testing.assert_allclose(out4[0], phidp_true0, atol=1.0, rtol=0.02)

        # check copy
        if copy:
            np.testing.assert_array_equal(in4, phidp_raw4)
        else:
            assert not np.array_equal(in4, phidp_raw4)

    def test_kdp_from_phidp_nan(self, derivation_method):
        if derivation_method == "lstsq" and sys.platform.startswith("win"):
            pytest.skip("fails on windows due to MKL issue")
        if derivation_method == "lstsq" and sys.platform.startswith("linux"):
            pytest.skip("segfaults on linux for some unknow reason")

        window = 7

        # intercompare with lanczos method without NaN-handling
        out0 = dp.kdp_from_phidp(
            self.phidp_true_nan.copy(),
            dr=self.dr,
            method="lanczos_conv",
            skipna=False,
            winlen=window,
        )
        outx = dp.kdp_from_phidp(
            self.phidp_true_nan.copy(),
            dr=self.dr,
            method=derivation_method,
            skipna=False,
            winlen=window,
        )
        np.testing.assert_array_almost_equal(outx, out0, decimal=4)

    def test_kdp_from_phidp(self, derivation_method):
        if derivation_method == "lstsq" and sys.platform.startswith("win"):
            pytest.skip("fails on windows due to MKL issue")
        if derivation_method == "lstsq" and sys.platform.startswith("linux"):
            pytest.skip("segfaults on linux for some unknow reason")

        window = 7

        # compare with true kdp
        out = dp.kdp_from_phidp(
            self.phidp_true.copy(), dr=self.dr, method=derivation_method, winlen=window
        )
        outx = out[:, self.pad : -self.pad]
        res = self.kdp_true[:, self.pad : -self.pad]
        np.testing.assert_array_almost_equal(outx, res, decimal=4)

        # intercompare with lanczos method with NaN handling
        out0 = dp.kdp_from_phidp(
            self.phidp_true.copy(), dr=self.dr, method="lanczos_conv", winlen=window
        )
        np.testing.assert_array_almost_equal(out, out0, decimal=4)

    def test_linear_despeckle(self, ndespeckle):
        dp.linear_despeckle(self.phidp_raw0, ndespeckle=ndespeckle, copy=True)

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
        dp._fill_sweep(self.phidp_raw0, kind="linear")


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

    def test_texture(self):
        tex = dp.texture(self.img)
        np.testing.assert_array_equal(tex[1:4, 1:4], self.pixel)
        np.testing.assert_array_equal(tex[4:7, 5:9], self.line)
        np.testing.assert_array_equal(tex[19:22], self.spike)
        np.testing.assert_array_equal(tex[59:121, 1:8], self.rainfield)


class TestDepolarization:
    def test_depolarization(self):
        zdr = np.linspace(-0.5, 0.5, 10)
        rho = np.linspace(0.0, 1.0, 10)

        dr_0 = [
            -12.719937,
            -12.746507,
            -12.766551,
            -12.779969,
            -12.786695,
            -12.786695,
            -12.779969,
            -12.766551,
            -12.746507,
            -12.719937,
        ]
        dr_1 = [
            0.0,
            -0.96266,
            -1.949568,
            -2.988849,
            -4.118078,
            -5.394812,
            -6.921361,
            -8.919312,
            -12.067837,
            -24.806473,
        ]

        np.testing.assert_array_almost_equal(dp.depolarization(zdr, 0.9), dr_0)
        np.testing.assert_array_almost_equal(dp.depolarization(1.0, rho), dr_1)
