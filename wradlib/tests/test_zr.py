#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.zr as zr
import wradlib.trafo as trafo
import numpy as np


class ZRConversionTest(unittest.TestCase):
    def setUp(self):
        img = np.zeros((5, 11), dtype=np.float32)
        img[0:1, 1:3] = 11.  # precip field
        img[0:1, 6:9] = 45.
        img[2:3, 1:3] = 38.
        img[2:3, 6:9] = 2.0
        img[3:4, 1:7] = 5.0

        self.img = img

    def test_z_to_r(self):
        self.assertEqual(zr.z_to_r(trafo.idecibel(10.)), 0.1537645610180688)

    def test_r_to_z(self):
        self.assertEqual(zr.r_to_z(0.15), 9.611164492610417)

    def test_z_to_r_enhanced(self):
        res_rr, res_si = zr.z_to_r_enhanced(trafo.idecibel(self.img))
        res_rr2, res_si2 = zr.z_to_r_enhanced(trafo.idecibel(self.img),
                                              algo='mdfilt', mode='mirror')
        res_rr3, res_si3 = zr.z_to_r_enhanced(trafo.idecibel(self.img),
                                              algo='mdcorr', xmode='mirror',
                                              ymode='mirror')

        rr = np.array([[3.64633237e-02, 1.77564547e-01, 1.77564547e-01,
                        3.17838962e-02, 3.17838962e-02, 1.62407903e-02,
                        2.37427600e+01, 2.37427600e+01, 2.37427600e+01,
                        1.62407903e-02, 3.17838962e-02],
                       [1.62407903e-02, 1.62407903e-02, 1.62407903e-02,
                        1.62407903e-02, 3.17838962e-02, 1.62407903e-02,
                        1.62407903e-02, 1.62407903e-02, 1.62407903e-02,
                        1.62407903e-02, 3.17838962e-02],
                       [1.62407903e-02, 8.64681611e+00, 8.64681611e+00,
                        1.62407903e-02, 3.17838962e-02, 3.17838962e-02,
                        4.41635812e-02, 4.41635812e-02, 4.41635812e-02,
                        3.17838962e-02, 3.17838962e-02],
                       [1.62407903e-02, 3.69615367e-02, 3.69615367e-02,
                        3.69615367e-02, 7.23352513e-02, 7.23352513e-02,
                        7.23352513e-02, 3.17838962e-02, 3.17838962e-02,
                        3.17838962e-02, 3.17838962e-02],
                       [3.64633237e-02, 3.64633237e-02, 3.64633237e-02,
                        3.17838962e-02, 3.17838962e-02, 1.62407903e-02,
                        1.62407903e-02, 1.62407903e-02, 1.62407903e-02,
                        1.62407903e-02, 3.17838962e-02]])
        si = np.array([[4.71428575, 4.58333337, 4.58333337, 2.75000002,
                        0., 11.25000003, -1., -1.,
                        -1., 11.25000003, 0.],
                       [13.99999989, 12.2499999, 12.2499999, 8.1666666,
                        0., 7.83333337, 11.75000005, 11.75000005,
                        11.75000005, 7.83333337, 0.],
                       [16.28571408, -1., -1., 9.91666655,
                        1.25000001, 1.41666669, 1.75000004, 1.50000004,
                        0.83333337, 0.50000002, 0.],
                       [11.57142844, 9.91666655, 10.33333322, 7.99999994,
                        2.50000003, 2.50000003, 2.25000003, 1.41666669,
                        0.50000002, 0.33333335, 0.],
                       [4.57142861, 4.00000004, 4.00000004, 3.08333336,
                        1.25000001, 8.75000003, 12.50000004, 12.08333337,
                        11.25000003, 7.50000002, 0.]])
        np.testing.assert_allclose(rr, res_rr)
        np.testing.assert_allclose(rr, res_rr2)
        np.testing.assert_allclose(rr, res_rr3)

        self.assertTrue(np.allclose(si, res_si))


if __name__ == '__main__':
    unittest.main()
