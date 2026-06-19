#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest
import wradlib_data
import xarray as xr

from wradlib import atten, io

from . import get_wradlib_data_file


@pytest.fixture
def att_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestAttenuation:
        gateset = np.arange(2 * 2 * 5).reshape((2, 2, 5)) * 3
        gateset_result = np.array(
            [
                [
                    [
                        0.00000000e00,
                        4.00000000e-04,
                        1.04876587e-03,
                        2.10105093e-03,
                        3.80794694e-03,
                    ],
                    [
                        0.00000000e00,
                        4.48807382e-03,
                        1.17721446e-02,
                        2.35994018e-02,
                        4.28175682e-02,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        5.03570165e-02,
                        1.32692110e-01,
                        2.68007888e-01,
                        4.92303379e-01,
                    ],
                    [
                        0.00000000e00,
                        5.65015018e-01,
                        1.56873147e00,
                        3.48241974e00,
                        7.70744561e00,
                    ],
                ],
            ]
        )

    yield TestAttenuation


def test_calc_attenuation_forward(att_data):
    """basic test for correct numbers"""
    a = 2e-4
    b = 0.7
    gate_length = 1.0
    result = atten.calc_attenuation_forward(
        att_data.gateset, a=a, b=b, gate_length=gate_length
    )
    assert np.allclose(result, att_data.gateset_result)


def test__sector_filter_1():
    """test sector filter with odd sector size"""
    mask = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=int)
    ref = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], dtype=int)
    min_sector_size = 3
    result = atten._sector_filter(mask, min_sector_size)
    np.testing.assert_equal(result, ref)


def test__sector_filter_2():
    """test sector filter with even sector size"""
    mask = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=int)
    ref = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    min_sector_size = 4
    result = atten._sector_filter(mask, min_sector_size)
    np.testing.assert_equal(result, ref)


def test_correct_attenuation_hb():
    filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
    filename = get_wradlib_data_file(filestr)
    gateset, attrs = io.read_dx(filename)
    atten.correct_attenuation_hb(gateset, mode="warn")
    atten.correct_attenuation_hb(gateset, mode="nan")
    atten.correct_attenuation_hb(gateset, mode="zero")
    with pytest.raises(atten.AttenuationOverflowError):
        atten.correct_attenuation_hb(gateset, mode="except")


def test_correct_attenuation_hb_xarray(dx_swp):
    dx_swp.wrl.atten.correct_attenuation_hb(mode="warn")
    dx_swp.wrl.atten.correct_attenuation_hb(mode="nan")
    dx_swp.wrl.atten.correct_attenuation_hb(mode="zero")
    with pytest.raises(atten.AttenuationOverflowError):
        dx_swp.wrl.atten.correct_attenuation_hb(mode="except")


def test_correct_attenuation_constrained():
    filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
    filename = get_wradlib_data_file(filestr)
    gateset, attrs = io.read_dx(filename)
    atten.correct_attenuation_constrained(gateset)


def test_correct_attenuation_constrained_xarray(dx_swp):
    dx_swp.wrl.atten.correct_attenuation_constrained()


def test_pia_from_kdp():
    filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
    filename = get_wradlib_data_file(filestr)
    gateset, attrs = io.read_dx(filename)
    atten.pia_from_kdp(gateset, 1)


def test_pia_from_kdp_xarray(dx_swp):
    dx_swp.wrl.atten.pia_from_kdp()


def test_correct_radome_attenuation_empirical(att_data):
    goodresult = np.array(
        [
            [
                [0.0114712, 0.0114712, 0.0114712, 0.0114712, 0.0114712],
                [0.0114712, 0.0114712, 0.0114712, 0.0114712, 0.0114712],
            ],
            [
                [0.86021834, 0.86021834, 0.86021834, 0.86021834, 0.86021834],
                [0.86021834, 0.86021834, 0.86021834, 0.86021834, 0.86021834],
            ],
        ]
    )
    result = atten.correct_radome_attenuation_empirical(att_data.gateset)
    assert np.allclose(result, goodresult)


def test_bisect_reference_attenuation(att_data):
    goodresult = np.array(
        [
            [
                [
                    0.00000000e00,
                    1.90300000e-04,
                    4.98939928e-04,
                    9.99520182e-04,
                    1.81143180e-03,
                ],
                [
                    0.00000000e00,
                    2.13520112e-03,
                    5.59928382e-03,
                    1.12205058e-02,
                    2.03453241e-02,
                ],
            ],
            [
                [
                    0.00000000e00,
                    2.39573506e-02,
                    6.29619483e-02,
                    1.26618942e-01,
                    2.30923218e-01,
                ],
                [
                    0.00000000e00,
                    4.34978358e-02,
                    1.12575637e-01,
                    2.22703609e-01,
                    3.99374943e-01,
                ],
            ],
        ]
    )
    goodamid = np.array(
        [[9.51500000e-05, 9.51500000e-05], [9.51500000e-05, 2.33043854e-05]]
    )
    goodb = np.array([[0.7, 0.7], [0.7, 0.66]])
    result, amid, b = atten.bisect_reference_attenuation(
        att_data.gateset, pia_ref=np.array([[0.0001, 0.01], [0.1, 0.2]])
    )
    assert np.allclose(result, goodresult)
    assert np.allclose(amid, goodamid)
    assert np.allclose(b, goodb)


@pytest.fixture
def simple_phidp_dbz():
    r = np.arange(200) * 100.0

    phidp = xr.DataArray(
        np.linspace(0, 30, 200),
        coords={"range": r},
        dims="range",
    )

    phidp[60:120] += 10.0
    phidp[150] = np.nan

    dbz = xr.DataArray(
        40 - 0.05 * np.arange(200),
        coords={"range": r},
        dims="range",
        name="DBZH",
    )

    return phidp, dbz


def test_specific_attenuation_output_structure(simple_phidp_dbz):
    phidp, dbz = simple_phidp_dbz

    out = atten.specific_attenuation_zphi(phidp, dbz, alpha=0.3, b=0.6, rng=2000.0)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("range",)

    assert "units" in out.attrs
    assert out.attrs["units"] == "dB/km"
    assert "standard_name" in out.attrs
    assert "long_name" in out.attrs
    assert "short_name" in out.attrs


def test_specific_attenuation_finite(simple_phidp_dbz):
    phidp, dbz = simple_phidp_dbz

    out = atten.specific_attenuation_zphi(phidp, dbz, alpha=0.3, b=0.6, rng=2000.0)

    assert np.all(np.isfinite(out.values))
    assert np.nanmin(out.values) >= 0


def test_specific_attenuation_alpha_broadcast(simple_phidp_dbz):
    phidp, dbz = simple_phidp_dbz

    out_scalar = atten.specific_attenuation_zphi(
        phidp, dbz, alpha=0.3, b=0.6, rng=2000.0
    )

    out_vector = atten.specific_attenuation_zphi(
        phidp, dbz, alpha=[0.2, 0.3], b=0.6, rng=2000.0
    )

    assert "alpha" in out_vector.dims
    assert out_vector.sizes["alpha"] == 2
    assert out_vector.sizes["range"] == out_scalar.sizes["range"]


def test_specific_attenuation_depends_on_phidp(simple_phidp_dbz):
    phidp, dbz = simple_phidp_dbz

    out1 = atten.specific_attenuation_zphi(phidp, dbz, alpha=0.3, b=0.6, rng=2000.0)

    phidp2 = phidp + 5.0  # perturb phase
    out2 = atten.specific_attenuation_zphi(phidp2, dbz, alpha=0.3, b=0.6, rng=2000.0)

    assert np.allclose(out1.values, out2.values)


def test_specific_attenuation_data():
    fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
    with xr.open_dataset(fname, engine="gamic", group="sweep_0") as swp:
        mask = swp.RHOHV > 0.9
        phidp = swp.PHIDP.where(mask)
        dbz = swp.DBZH.where(mask)

    out1 = atten.specific_attenuation_zphi(phidp, dbz, 0.28, 0.78, rng=2000.0)
    out2 = phidp.wrl.atten.specific_attenuation_zphi(dbz, 0.28, 0.78, rng=2000.0)

    out1 = out1.max("range").sel(azimuth=slice(100, 150))
    out2 = out2.max("range").sel(azimuth=slice(100, 150))
    wanted = np.array(
        [
            1.07407273,
            0.50883621,
            0.5075296,
            0.70262613,
            1.17170939,
            1.04093076,
            0.6277884,
            0.40821078,
            0.87730694,
            1.09753143,
            1.15603688,
            0.9731284,
            3.14065076,
            1.39616165,
            1.28489366,
            1.62729008,
            1.65306511,
            1.10869691,
            1.77924222,
            2.24248579,
            0.58178274,
            2.52007878,
            1.71641173,
            1.13451356,
            0.36279973,
            0.54705892,
            0.5706312,
            0.75613272,
            0.82019211,
            1.41853458,
            1.36950221,
            1.31224489,
            1.57759488,
            1.61514931,
            2.13054111,
            1.25734168,
            1.26039325,
            1.00678335,
            0.96663833,
            0.48267999,
            0.69075709,
            1.28177211,
            1.21027243,
            1.45197661,
            1.90574993,
            2.31374573,
            2.82394853,
            1.99345444,
            1.50467872,
            1.16656558,
        ]
    )
    np.testing.assert_allclose(out1.values, wanted, atol=1e-6)
    np.testing.assert_allclose(out2.values, wanted, atol=1e-6)
