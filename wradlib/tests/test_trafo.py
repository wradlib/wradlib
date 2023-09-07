#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import trafo


@pytest.fixture
def tdata():
    @dataclass(init=False, repr=False, eq=False)
    class TestTransformation:
        rvp = np.array([0.0, 128.0, 255.0])
        dbz = np.array([-32.5, 31.5, 95.0])
        lin = np.array([1e-4, 1.0, 1e4])
        dec = np.array([-40.0, 0.0, 40.0])
        r = np.array([5.0, 10.0, 20.0])
        kdp = np.array([0.0, 1.0, 2.0, 5.0])
        # speed in m/s
        speedsi = np.array([0.0, 1.0, 50.0])
        # speed in km/h
        speedkmh = np.array([0.0, 3.6, 180.0])
        # speed in miles/h
        speedmph = np.array([0.0, 2.23693629, 111.8468146])
        # speed in knots
        speedkts = np.array([0.0, 1.94384449, 97.19222462])

    yield TestTransformation


def test_rvp_to_dbz(tdata):
    assert np.allclose(trafo.rvp_to_dbz(tdata.rvp), tdata.dbz)


def test_decibel(tdata):
    assert np.allclose(trafo.decibel(tdata.lin), tdata.dec)


def test_idecibel(tdata):
    assert np.allclose(trafo.idecibel(tdata.dec), tdata.lin)


def test_r_to_depth(tdata):
    assert np.allclose(trafo.r_to_depth(tdata.r, 720), np.array([1.0, 2.0, 4.0]))
    assert np.allclose(trafo.r_to_depth(tdata.r, 360), np.array([0.5, 1.0, 2.0]))


def test_kdp_tp_r(tdata):
    assert np.allclose(
        trafo.kdp_to_r(tdata.kdp, 9.45),
        np.array([0.0, 19.11933017, 34.46261032, 75.09260608]),
    )


def test_si_to_kmh(tdata):
    assert np.allclose(trafo.si_to_kmh(tdata.speedsi), tdata.speedkmh)


def test_si_to_mph(tdata):
    assert np.allclose(trafo.si_to_mph(tdata.speedsi), tdata.speedmph)


def test_si_to_kts(tdata):
    assert np.allclose(trafo.si_to_kts(tdata.speedsi), tdata.speedkts)


def test_kmh_to_si(tdata):
    assert np.allclose(trafo.kmh_to_si(tdata.speedkmh), tdata.speedsi)


def test_mph_to_si(tdata):
    assert np.allclose(trafo.mph_to_si(tdata.speedmph), tdata.speedsi)


def test_kts_to_si(tdata):
    assert np.allclose(trafo.kts_to_si(tdata.speedkts), tdata.speedsi)
