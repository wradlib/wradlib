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
import numpy as np
import wradlib.trafo as trafo


def test__rvp2dBZ():
    assert np.allclose(trafo.rvp2dBZ(np.array([0.,128.,255.])), np.array([-32.5,31.5,95.0]))


def test__decibel():
    assert np.allclose(trafo.decibel(np.array([1e-4, 1, 1e4])), np.array([-40,0,40]))


def test__idecibel():
    assert np.allclose(trafo.idecibel(np.array([-40, 0, 40])), np.array([1e-4, 1, 1e4]))


def test__r2depth():
    assert np.allclose(trafo.r2depth(np.array([5., 10., 20.]),720), np.array([1.,2.,4.]))
    assert np.allclose(trafo.r2depth(np.array([5., 10., 20.]),360), np.array([0.5,1.,2.]))