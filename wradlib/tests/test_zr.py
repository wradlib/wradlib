#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import trafo, zr


@pytest.fixture
def zr_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestZRConversion:
        img = np.zeros((5, 11), dtype=np.float32)
        img[0:1, 1:3] = 11.0  # precip field
        img[0:1, 6:9] = 45.0
        img[2:3, 1:3] = 38.0
        img[2:3, 6:9] = 2.0
        img[3:4, 1:7] = 5.0

        img = np.stack([img, img * 1.1, img * 1.2], axis=0)

    yield TestZRConversion


def test_z_to_r():
    assert zr.z_to_r(trafo.idecibel(10.0)) == 0.1537645610180688


def test_r_to_z():
    assert zr.r_to_z(0.15) == 9.611164492610417


def test_z_to_r_enhanced(zr_data):
    res_rr, res_si = zr.z_to_r_enhanced(trafo.idecibel(zr_data.img), polar=True)
    res_rr2, res_si2 = zr.z_to_r_enhanced(trafo.idecibel(zr_data.img[0]), polar=True)

    rr = np.array(
        [
            [
                3.64633237e-02,
                1.77564547e-01,
                1.77564547e-01,
                3.17838962e-02,
                3.17838962e-02,
                1.62407903e-02,
                2.37427600e01,
                2.37427600e01,
                2.37427600e01,
                1.62407903e-02,
                3.17838962e-02,
            ],
            [
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                3.17838962e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                3.17838962e-02,
            ],
            [
                1.62407903e-02,
                8.64681611e00,
                8.64681611e00,
                1.62407903e-02,
                3.17838962e-02,
                3.17838962e-02,
                4.41635812e-02,
                4.41635812e-02,
                4.41635812e-02,
                3.17838962e-02,
                3.17838962e-02,
            ],
            [
                1.62407903e-02,
                3.69615367e-02,
                3.69615367e-02,
                3.69615367e-02,
                7.23352513e-02,
                7.23352513e-02,
                7.23352513e-02,
                3.17838962e-02,
                3.17838962e-02,
                3.17838962e-02,
                3.17838962e-02,
            ],
            [
                3.64633237e-02,
                3.64633237e-02,
                3.64633237e-02,
                3.17838962e-02,
                3.17838962e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                1.62407903e-02,
                3.17838962e-02,
            ],
        ]
    )
    si = np.array(
        [
            [
                4.71428575,
                4.58333337,
                4.58333337,
                2.75000002,
                0.0,
                11.25000003,
                -1.0,
                -1.0,
                -1.0,
                11.25000003,
                0.0,
            ],
            [
                13.99999989,
                12.2499999,
                12.2499999,
                8.1666666,
                0.0,
                7.83333337,
                11.75000005,
                11.75000005,
                11.75000005,
                7.83333337,
                0.0,
            ],
            [
                16.28571408,
                -1.0,
                -1.0,
                9.91666655,
                1.25000001,
                1.41666669,
                1.75000004,
                1.50000004,
                0.83333337,
                0.50000002,
                0.0,
            ],
            [
                11.57142844,
                9.91666655,
                10.33333322,
                7.99999994,
                2.50000003,
                2.50000003,
                2.25000003,
                1.41666669,
                0.50000002,
                0.33333335,
                0.0,
            ],
            [
                4.57142861,
                4.00000004,
                4.00000004,
                3.08333336,
                1.25000001,
                8.75000003,
                12.50000004,
                12.08333337,
                11.25000003,
                7.50000002,
                0.0,
            ],
        ]
    )

    np.testing.assert_almost_equal(si, res_si[0], decimal=6)
    np.testing.assert_array_almost_equal(rr, res_rr[0], decimal=6)
    np.testing.assert_almost_equal(si, res_si2, decimal=6)
    np.testing.assert_array_almost_equal(rr, res_rr2, decimal=6)
