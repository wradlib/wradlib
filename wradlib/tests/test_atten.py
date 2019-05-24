#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.atten as atten
import unittest
from .. import util as util
from .. import io as io


class TestAttenuation(unittest.TestCase):
    def setUp(self):
        self.gateset = np.arange(2 * 2 * 5).reshape((2, 2, 5)) * 3
        self.gateset_result = np.array([[[0.00000000e+00, 4.00000000e-04,
                                          1.04876587e-03, 2.10105093e-03,
                                          3.80794694e-03],
                                         [0.00000000e+00, 4.48807382e-03,
                                          1.17721446e-02, 2.35994018e-02,
                                          4.28175682e-02]],
                                        [[0.00000000e+00, 5.03570165e-02,
                                          1.32692110e-01, 2.68007888e-01,
                                          4.92303379e-01],
                                         [0.00000000e+00, 5.65015018e-01,
                                          1.56873147e+00, 3.48241974e+00,
                                          7.70744561e+00]]])

    def test_calc_attenuation_forward(self):
        """basic test for correct numbers"""
        a = 2e-4
        b = 0.7
        gate_length = 1.
        result = atten.calc_attenuation_forward(self.gateset, a, b,
                                                gate_length)
        self.assertTrue(np.allclose(result, self.gateset_result))

    # def test__sector_filter_1(self):
    #     # """test sector filter with odd sector size"""
    #     mask = np.array([1,1,0,1,0,1,1,0,1,1,1,0,1], dtype=np.int)
    #     ref =  np.array([1,1,0,0,0,0,0,0,1,1,1,0,1], dtype=np.int)
    #     min_sector_size = 3
    #     result = atten._sector_filter(mask, min_sector_size)
    #     print(result)
    #     print(ref)
    #     self.assertTrue(np.all(result == ref))
    #     #pass

    # def test__sector_filter_2(self):
    #     """test sector filter with even sector size"""
    #     mask = np.array([1,1,1,0,1,0,1,1,0,1,1,1,1,0,1], dtype=np.int)
    #     ref =  np.array([1,1,1,0,0,0,0,0,0,1,1,1,1,0,1], dtype=np.int)
    #     min_sector_size = 4
    #     result = atten._sector_filter(mask, min_sector_size)
    #     print(result)
    #     print(ref)
    #     self.assertTrue(np.all(result == ref))
    #     #pass

    def test_correct_attenuation_hb(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.read_dx(filename)
        atten.correct_attenuation_hb(gateset, mode='warn')
        atten.correct_attenuation_hb(gateset, mode='nan')
        atten.correct_attenuation_hb(gateset, mode='zero')
        with self.assertRaises(atten.AttenuationOverflowError):
            atten.correct_attenuation_hb(gateset, mode='except')

    def test_correct_attenuation_constrained(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.read_dx(filename)
        atten.correct_attenuation_constrained(gateset)

    def test_correct_radome_attenuation_empirical(self):
        goodresult = np.array([[[0.0114712, 0.0114712, 0.0114712, 0.0114712,
                                 0.0114712],
                                [0.0114712, 0.0114712, 0.0114712, 0.0114712,
                                 0.0114712]],
                               [[0.86021834, 0.86021834, 0.86021834,
                                 0.86021834, 0.86021834],
                                [0.86021834, 0.86021834, 0.86021834,
                                 0.86021834, 0.86021834]]])
        result = atten.correct_radome_attenuation_empirical(self.gateset)
        self.assertTrue(np.allclose(result, goodresult))

    def test_bisect_reference_attenuation(self):
        goodresult = np.array([[[0.00000000e+00, 1.90300000e-04,
                                 4.98939928e-04, 9.99520182e-04,
                                 1.81143180e-03],
                                [0.00000000e+00, 2.13520112e-03,
                                 5.59928382e-03, 1.12205058e-02,
                                 2.03453241e-02]],
                               [[0.00000000e+00, 2.39573506e-02,
                                 6.29619483e-02, 1.26618942e-01,
                                 2.30923218e-01],
                                [0.00000000e+00, 4.34978358e-02,
                                 1.12575637e-01, 2.22703609e-01,
                                 3.99374943e-01]]])
        goodamid = np.array([[9.51500000e-05, 9.51500000e-05],
                             [9.51500000e-05, 2.33043854e-05]])
        goodb = np.array([[0.7, 0.7], [0.7, 0.66]])
        result, amid, b = atten.bisect_reference_attenuation(self.gateset,
                                                             pia_ref=np.array(
                                                                 [[0.0001,
                                                                   0.01],
                                                                  [0.1, 0.2]]))
        self.assertTrue(np.allclose(result, goodresult))
        self.assertTrue(np.allclose(amid, goodamid))
        self.assertTrue(np.allclose(b, goodb))


if __name__ == '__main__':
    unittest.main()
