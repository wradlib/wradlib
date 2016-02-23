# -------------------------------------------------------------------------------
# Name:        test_trafo
# Purpose:     testing file for the wradlib.trafo module
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     12.03.2013
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
# -------------------------------------------------------------------------------
# !/usr/bin/env python
import unittest
import numpy as np
import wradlib.trafo as trafo


class TransformationTest(unittest.TestCase):
    def setUp(self):
        self.rvp = np.array([0., 128., 255.])
        self.dbz = np.array([-32.5, 31.5, 95.0])
        self.lin = np.array([1e-4, 1, 1e4])
        self.dec = np.array([-40, 0, 40])
        self.r = np.array([5., 10., 20.])
        self.speedsi = np.array([0., 1., 50.])  # speed in m/s
        self.speedkmh = np.array([0., 3.6, 180.])  # speed in km/h
        self.speedmph = np.array([0., 2.23693629, 111.8468146])  # speed in miles/h
        self.speedkts = np.array([0., 1.94384449, 97.19222462])  # speed in knots

    def test_rvp2dBZ(self):
        self.assertTrue(np.allclose(trafo.rvp2dBZ(self.rvp), self.dbz))

    def test_decibel(self):
        self.assertTrue(np.allclose(trafo.decibel(self.lin), self.dec))

    def test_idecibel(self):
        self.assertTrue(np.allclose(trafo.idecibel(self.dec), self.lin))

    def test_r2depth(self):
        self.assertTrue(np.allclose(trafo.r2depth(self.r, 720), np.array([1., 2., 4.])))
        self.assertTrue(np.allclose(trafo.r2depth(self.r, 360), np.array([0.5, 1., 2.])))

    def test_si2kmh(self):
        self.assertTrue(np.allclose(trafo.si2kmh(self.speedsi), self.speedkmh))

    def test_si2mph(self):
        self.assertTrue(np.allclose(trafo.si2mph(self.speedsi), self.speedmph))

    def test_si2kts(self):
        self.assertTrue(np.allclose(trafo.si2kts(self.speedsi), self.speedkts))

    def test_kmh2si(self):
        self.assertTrue(np.allclose(trafo.kmh2si(self.speedkmh), self.speedsi))

    def test_mph2si(self):
        self.assertTrue(np.allclose(trafo.mph2si(self.speedmph), self.speedsi))

    def test_kts2si(self):
        self.assertTrue(np.allclose(trafo.kts2si(self.speedkts), self.speedsi))


if __name__ == '__main__':
    unittest.main()
