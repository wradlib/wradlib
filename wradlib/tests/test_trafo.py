#-------------------------------------------------------------------------------
# Name:        test_trafo
# Purpose:     testing file for the wradlib.trafo module
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     12.03.2013
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import unittest
import numpy as np
import wradlib.trafo as trafo


class TransformationTest(unittest.TestCase):

    def setUp(self):
        self.rvp = np.array([0.,128.,255.])
        self.dbz = np.array([-32.5,31.5,95.0])
        self.lin = np.array([1e-4, 1, 1e4])
        self.dec = np.array([-40,0,40])
        self.r = np.array([5., 10., 20.])

    def test_rvp2dBZ(self):
        self.assertTrue(np.allclose(trafo.rvp2dBZ(self.rvp), self.dbz))

    def test_decibel(self):
        self.assertTrue(np.allclose(trafo.decibel(self.lin), self.dec))

    def test_idecibel(self):
        self.assertTrue(np.allclose(trafo.idecibel(self.dec), self.lin))

    def test_r2depth(self):
        self.assertTrue(np.allclose(trafo.r2depth(self.r,720), np.array([1.,2.,4.])))
        self.assertTrue(np.allclose(trafo.r2depth(self.r,360), np.array([0.5,1.,2.])))

if __name__ == '__main__':
    unittest.main()