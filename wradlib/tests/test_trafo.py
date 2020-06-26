#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np

from wradlib import trafo


class TestTransformation:
    rvp = np.array([0.0, 128.0, 255.0])
    dbz = np.array([-32.5, 31.5, 95.0])
    lin = np.array([1e-4, 1, 1e4])
    dec = np.array([-40, 0, 40])
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

    def test_rvp_to_dbz(self):
        assert np.allclose(trafo.rvp_to_dbz(self.rvp), self.dbz)

    def test_decibel(self):
        assert np.allclose(trafo.decibel(self.lin), self.dec)

    def test_idecibel(self):
        assert np.allclose(trafo.idecibel(self.dec), self.lin)

    def test_r_to_depth(self):
        assert np.allclose(trafo.r_to_depth(self.r, 720), np.array([1.0, 2.0, 4.0]))
        assert np.allclose(trafo.r_to_depth(self.r, 360), np.array([0.5, 1.0, 2.0]))

    def test_kdp_tp_r(self):
        assert np.allclose(
            trafo.kdp_to_r(self.kdp, 9.45),
            np.array([0.0, 19.11933017, 34.46261032, 75.09260608]),
        )

    def test_si_to_kmh(self):
        assert np.allclose(trafo.si_to_kmh(self.speedsi), self.speedkmh)

    def test_si_to_mph(self):
        assert np.allclose(trafo.si_to_mph(self.speedsi), self.speedmph)

    def test_si_to_kts(self):
        assert np.allclose(trafo.si_2_kts(self.speedsi), self.speedkts)

    def test_kmh_to_si(self):
        assert np.allclose(trafo.kmh_to_si(self.speedkmh), self.speedsi)

    def test_mph_to_si(self):
        assert np.allclose(trafo.mph_to_si(self.speedmph), self.speedsi)

    def test_kts_to_si(self):
        assert np.allclose(trafo.kts_to_si(self.speedkts), self.speedsi)
