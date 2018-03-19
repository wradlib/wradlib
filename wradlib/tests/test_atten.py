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

    def test_correctAttenuationHB(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.readDX(filename)
        atten.correctAttenuationHB(gateset, mode='warn')
        atten.correctAttenuationHB(gateset, mode='nan')
        atten.correctAttenuationHB(gateset, mode='zero')
        self.assertRaises(atten.AttenuationOverflowError,
                          lambda: atten.correctAttenuationHB(gateset,
                                                             mode='except'))

    def test_correctAttenuationKraemer(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.readDX(filename)
        atten.correctAttenuationKraemer(gateset)
        atten.correctAttenuationKraemer(gateset, mode='warn')
        atten.correctAttenuationKraemer(gateset, mode='nan')
        # testfunc = atten.correctAttenuationKraemer
        # self.assertRaises(atten.AttenuationOverflowError,
        #                   lambda: testfunc(gateset,
        #                                    mode='except'))

    def test_correctAttenuationHJ(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.readDX(filename)
        atten.correctAttenuationHJ(gateset, a_max=4.565e-5, b=0.73125, n=1,
                                   mode='cap', thrs_dBZ=100.0,
                                   max_PIA=4.82)
        atten.correctAttenuationHJ(gateset, a_max=4.565e-5,
                                   b=0.73125, n=1, mode='warn',
                                   thrs_dBZ=100.0, max_PIA=4.82)
        atten.correctAttenuationHJ(gateset,  a_max=4.565e-5,
                                   b=0.73125, n=1, mode='nan',
                                   thrs_dBZ=100.0, max_PIA=4.82)
        atten.correctAttenuationHJ(gateset,  a_max=4.565e-5,
                                   b=0.73125, n=1, mode='zero',
                                   thrs_dBZ=100.0, max_PIA=4.82)
        self.assertRaises(atten.AttenuationOverflowError,
                          lambda: atten.correctAttenuationHJ(gateset,
                                                             a_max=4.565e-5,
                                                             b=0.73125, n=1,
                                                             mode='except',
                                                             thrs_dBZ=100.0,
                                                             max_PIA=4.82))

    def test_correctAttenuationConstrainer(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.readDX(filename)
        atten.correctAttenuationConstrained(gateset)
        atten.correctAttenuationConstrained(gateset, mode='warn')
        atten.correctAttenuationConstrained(gateset, mode='nan')
        atten.correctAttenuationConstrained(gateset, mode='zero')

    def test_correctAttenuationConstrained2(self):
        filestr = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        filename = util.get_wradlib_data_file(filestr)
        gateset, attrs = io.readDX(filename)
        atten.correctAttenuationConstrained2(gateset)


if __name__ == '__main__':
    unittest.main()
