#!/usr/bin/env python
# Copyright (c) 2011-2025, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest
import wradlib_data
import xarray as xr
from scipy import integrate

from wradlib import dp, util


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


@pytest.fixture
def dp_data():
    @dataclass(init=False, repr=False, eq=False)
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
        kdp_true = np.arange(az * rng, dtype=np.float64).reshape(az, rng)
        phidp_true = np.power(kdp_true, 2)
        dr = 0.1
        kdp_true /= dr
        phidp_true_nan = phidp_true.copy()
        phidp_true_nan[:, window:-1:10] = np.nan

    yield TestKDPFromPHIDP


def test_phidp_kdp_vulpiani(derivation_method, window, copy):
    # Todo: move data setup into fixture
    np.random.seed(42000)
    # Synthetic truth
    dr = 0.5
    r = np.arange(0, 500, dr)

    kdp_true0 = np.sin(0.3 * r)
    kdp_true0[kdp_true0 < 0] = 0.0
    phidp_true0 = 2 * integrate.cumulative_trapezoid(
        kdp_true0, axis=-1, initial=0, dx=dr
    )
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
    out0 = dp.phidp_kdp_vulpiani(
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

    out1 = dp.phidp_kdp_vulpiani(
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

    out2 = dp.phidp_kdp_vulpiani(
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

    out3 = dp.phidp_kdp_vulpiani(
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
    out4 = dp.phidp_kdp_vulpiani(
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


def test_kdp_from_phidp_nan(dp_data, derivation_method):
    window = 7

    # intercompare with lanczos method without NaN-handling
    out0 = dp.kdp_from_phidp(
        dp_data.phidp_true_nan.copy(),
        dr=dp_data.dr,
        method="lanczos_conv",
        skipna=False,
        winlen=window,
    )
    outx = dp.kdp_from_phidp(
        dp_data.phidp_true_nan.copy(),
        dr=dp_data.dr,
        method=derivation_method,
        skipna=False,
        winlen=window,
    )
    np.testing.assert_array_almost_equal(outx, out0, decimal=4)


def test_kdp_from_phidp(dp_data, derivation_method):
    window = 7

    # compare with true kdp
    out = dp.kdp_from_phidp(
        dp_data.phidp_true.copy(),
        dr=dp_data.dr,
        method=derivation_method,
        winlen=window,
    )
    outx = out[:, dp_data.pad : -dp_data.pad]
    res = dp_data.kdp_true[:, dp_data.pad : -dp_data.pad]
    np.testing.assert_array_almost_equal(outx, res, decimal=4)

    # intercompare with lanczos method with NaN handling
    out0 = dp.kdp_from_phidp(
        dp_data.phidp_true.copy(), dr=dp_data.dr, method="lanczos_conv", winlen=window
    )
    np.testing.assert_array_almost_equal(out, out0, decimal=4)


def test_linear_despeckle(dp_data, ndespeckle):
    util.despeckle(dp_data.phidp_raw0, n=ndespeckle, copy=True)


def test_unfold_phi(dp_data):
    dp.unfold_phi(dp_data.phidp_raw0, dp_data.rho)
    dp.unfold_phi(dp_data.phidp_raw0, dp_data.rho, copy=True)

    # sanity check, unfold single ray
    phi_true = np.arange(540, dtype="float32")
    phi_true -= 180.0
    phi_raw1 = phi_true.copy()
    phi_raw1[phi_raw1 > 180] -= 360
    rho = np.ones(540, dtype="float32")
    out1 = dp.unfold_phi(phi_raw1, rho, copy=True)
    np.testing.assert_array_equal(out1, phi_true)


def test_unfold_phi_vulpiani():
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


def test__fill_sweep(dp_data):
    dp._fill_sweep(dp_data.phidp_raw0, kind="linear")


def test_texture_deprecation():
    with pytest.warns(DeprecationWarning):
        data = np.zeros((360, 1000))
        dp.texture(data)


def test_depolarization():
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


def test_depolarization_xarray(gamic_swp):
    gamic_swp.wrl.dp.depolarization(zdr="ZDR", rho="RHOHV")


def test_kdp_from_phidp_xarray(gamic_swp):
    gamic_swp.PHIDP.wrl.dp.kdp_from_phidp()


def test_phidp_kdp_vulpiani_xarray(gamic_swp):
    gamic_swp.PHIDP.wrl.dp.phidp_kdp_vulpiani()


def test_unfold_phi_xarray(gamic_swp):
    gamic_swp.wrl.dp.unfold_phi(phidp="PHIDP", rho="RHOHV")


def test_unfold_phi_vulpiani_xarray(gamic_swp):
    gamic_swp.wrl.dp.unfold_phi_vulpiani(phidp="PHIDP", kdp="KDP")


def test_system_phidp():
    phidp = xr.DataArray(
        [
            np.nan,
            np.nan,
            128.0,
            128.5,
            129.0,
            np.nan,
            130.0,
            130.5,
            np.nan,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            np.nan,
            np.nan,
        ],
        dims=["range"],
        coords={"range": np.arange(20) * 250.0},
        name="PHIDP",
    ).expand_dims(azimuth=[0])

    rng = 2000.0

    res_block = dp.system_phidp_block(
        phidp,
        rng=rng,
        n_lowest_rays=1,
    )

    res_window = dp.system_phidp_window(
        phidp,
        rng=rng,
        n_lowest_rays=1,
    )

    res_first = dp.system_phidp_first(
        phidp,
        n_valid_bins=9,
        n_lowest_rays=1,
    )

    res_hist = dp.system_phidp_hist(
        phidp,
        n_lowest_rays=1,
    )

    assert res_block["sysphi_ray"].item() == 132.75
    assert res_block["sysphi"].item() == 132.75
    assert res_window["sysphi_ray"].item() == 132.75
    assert res_window["sysphi"].item() == 132.75
    assert res_first["sysphi_ray"].item() == 130.5
    assert res_first["sysphi"].item() == 130.5
    assert res_hist["sysphi_peak"].item() == pytest.approx(128.55)
    assert res_hist["sysphi_first"].item() == pytest.approx(128.05)


def test_system_phidp_xarray():
    fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
    with xr.open_dataset(fname, engine="gamic", group="sweep_0") as swp:
        phase = swp.PHIDP.where(swp.RHOHV > 0.9)
    res_block = phase.wrl.dp.system_phidp_block(rng=2000.0)
    res_window = phase.wrl.dp.system_phidp_window(rng=2000.0)
    res_first = phase.wrl.dp.system_phidp_first()
    res_hist = phase.wrl.dp.system_phidp_hist()

    assert res_block["sysphi"].item() == pytest.approx(-80.149078)
    assert res_window["sysphi"].item() == pytest.approx(-80.348213)
    assert res_first["sysphi"].item() == pytest.approx(-80.651726)
    assert res_hist["sysphi_peak"].item() == pytest.approx(-80.55)
    assert res_hist["sysphi_first"].item() == pytest.approx(-81.6)


def test_rhohv_noise_correction_numpy():
    rho = np.array([0.95, 0.98, 0.99])
    snr = np.array([0.0, 10.0, 20.0])

    expected = rho * np.sqrt(1.0 + 1.0 / 10.0 ** (snr * 0.1))

    result = dp.rhohv_noise_correction(rho, snr)

    np.testing.assert_equal(result, expected)


def test_rhohv_noise_correction_xarray():
    rho = xr.DataArray(
        np.array([[0.95, 0.98], [0.96, 0.99]]),
        dims=("azimuth", "range"),
        coords={
            "azimuth": [0, 1],
            "range": [1000.0, 2000.0],
        },
        name="RHOHV",
    )

    snr = xr.DataArray(
        np.array([[10.0, 20.0], [15.0, 25.0]]),
        dims=("azimuth", "range"),
        coords=rho.coords,
    )

    result = dp.rhohv_noise_correction(rho, snr)

    expected = xr.DataArray(
        rho.values * np.sqrt(1.0 + 1.0 / 10.0 ** (snr.values * 0.1)),
        dims=rho.dims,
        coords=rho.coords,
    )

    xr.testing.assert_allclose(result, expected)

    assert result.name.endswith("_NC")


def make_phi():
    r = np.arange(20)

    data = np.full(20, np.nan)
    data[6:14] = np.arange(8)

    return xr.DataArray(
        data,
        coords={"range": r},
        dims="range",
    )


def test_delta_phidp_basic_properties():
    phi = make_phi()

    out = dp.delta_phidp(phi, rng=5.0)

    assert isinstance(out, xr.Dataset)

    for key in [
        "phib",
        "start_range",
        "stop_range",
        "first",
        "first_idx",
        "last",
        "last_idx",
        "dphi",
        "center_span",
    ]:
        assert key in out

    assert np.isfinite(out["start_range"].item())
    assert np.isfinite(out["stop_range"].item())

    np.testing.assert_allclose(
        out["dphi"],
        out["last"] - out["first"],
    )


def test_delta_phidp_finds_first_and_last_densest_windows():
    phi = make_phi()

    out = dp.delta_phidp(phi, rng=5.0)

    assert out["start_range"].item() == 6
    assert out["first_idx"].item() == 6

    assert out["stop_range"].item() == 13
    assert out["last_idx"].item() == 13

    assert out["center_span"].item() == 4


def test_delta_phidp():
    fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
    with xr.open_dataset(fname, engine="gamic", group="sweep_0") as swp:
        phase = swp.PHIDP.where(swp.RHOHV > 0.9)

    out = dp.delta_phidp(phase, rng=2000.0)
    np.testing.assert_allclose(out["first"][0].values, -78.62331)
    np.testing.assert_allclose(out["first_idx"][0].values, 44)
    np.testing.assert_allclose(out["last"][0].values, -78.203064)
    np.testing.assert_allclose(out["last_idx"][0].values, 569)
    np.testing.assert_allclose(out["dphi"][0].values, 0.420242, rtol=1e-6)


def test_delta_phidp_xarray():
    fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
    with xr.open_dataset(fname, engine="gamic", group="sweep_0") as swp:
        phase = swp.PHIDP.where(swp.RHOHV > 0.9)

    out = phase.wrl.dp.delta_phidp(rng=2000.0)
    np.testing.assert_allclose(out["first"][0].values, -78.62331)
    np.testing.assert_allclose(out["first_idx"][0].values, 44)
    np.testing.assert_allclose(out["last"][0].values, -78.203064)
    np.testing.assert_allclose(out["last_idx"][0].values, 569)
    np.testing.assert_allclose(out["dphi"][0].values, 0.420242, rtol=1e-6)
