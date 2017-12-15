#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.atten as atten
import unittest


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

    def test_sector_filter_1(self):
        # """test sector filter with odd sector size"""
        # mask = np.array([1,1,0,1,0,1,1,0,1,1,1,0,1], dtype=np.int)
        # ref =  np.array([1,1,0,0,0,0,0,0,1,1,1,0,1], dtype=np.int)
        # min_sector_size = 3
        # result = atten.sector_filter(mask, min_sector_size)
        # self.assertTrue(np.all(result == ref))
        pass

    def test_sector_filter_2(self):
        # """test sector filter with even sector size"""
        # mask = np.array([1,1,1,0,1,0,1,1,0,1,1,1,1,0,1], dtype=np.int)
        # ref =  np.array([1,1,1,0,0,0,0,0,0,1,1,1,1,0,1], dtype=np.int)
        # min_sector_size = 4
        # result = atten. sector_filter(mask, min_sector_size)
        # self.assertTrue(np.all(result == ref))
        pass

    def test_correctAttenuationConstrained2(self):
        # gateset = get_gateset()
        # a_max = 1.67e-4
        # a_min = 2.33e-5
        # na = 10
        # b_max = 0.7
        # b_min = 0.2
        # nb = 5
        # l = 1
        # constraints = [atten.constraint_dBZ]
        # constr_args = [[59.]]
        # thr_sec = 2
        # result = atten.correctAttenuationConstrained2(gateset,
        #                                            a_max,
        #                                            a_min,
        #                                            na,
        #                                            b_max,
        #                                            b_min,
        #                                            nb,
        #                                            l,
        #                                            mode='error',
        #                                            constraints=constraints,
        #                                            constr_args=constr_args,
        #                                            thr_sec=thr_sec,
        #                                            )
        pass


if __name__ == '__main__':
    unittest.main()
