#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.qual as qual
import unittest


class HelperFunctionsTest(unittest.TestCase):
    def test_beam_height_ft(self):
        self.assertTrue(np.allclose(qual.beam_height_ft(np.array([100, 200]),
                                                        np.array([2.0])),
                                    np.array([3.49053756, 6.98225089])))
        self.assertTrue(np.allclose(qual.beam_height_ft(np.array([100, 200]),
                                                        np.deg2rad([2.0]),
                                                        degrees=False),
                                    np.array([3.49053756, 6.98225089])))

    def test_beamheight_ft_doviak(self):
        self.assertTrue(
            np.allclose(qual.beam_height_ft_doviak(np.array([100, 200]),
                                                   np.array([2.0])),
                        np.array([3.49053756, 6.98225089])))
        self.assertTrue(
            np.allclose(qual.beam_height_ft_doviak(np.array([100, 200]),
                                                   np.deg2rad([2.0]),
                                                   degrees=False),
                        np.array([3.49053756, 6.98225089])))

    def test_get_bb_ratio(self):
        heights = np.array([[1100, 1100], [1100, 1100],
                            [1100, 1100], [1100, 1100]], dtype=np.float)
        widths = np.array([[50, 50], [50, 50], [50, 50], [50, 50]],
                          dtype=np.float)
        quality = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=np.float)
        print(quality.shape, quality, quality.T)
        z = np.array([[1075, 1100, 1125, 900], [1000, 1090, 1110, 1300]],
                     dtype=np.float)
        ratio_out = np.array([[[0., 0.5, 1., -3.5],
                               [-1.5, 0.3, 0.7, 4.5]],
                              [[0., 0.5, 1., -3.5],
                               [-1.5, 0.3, 0.7, 4.5]],
                              [[0., 0.5, 1., -3.5],
                               [-1.5, 0.3, 0.7, 4.5]],
                              [[0., 0.5, 1., -3.5],
                               [-1.5, 0.3, 0.7, 4.5]]])
        index_out = np.array([[True, True],
                              [True, True],
                              [True, False],
                              [False, True]])
        ratio, index = qual.get_bb_ratio(heights, widths, quality, z)
        np.testing.assert_array_equal(ratio, ratio_out)
        np.testing.assert_array_equal(index, index_out)


class BeamBlockFracTest(unittest.TestCase):
    def setUp(self):
        """
        create linear arrays of beamheight, beamradius and terrainheight

        """

        self.NBINS = 50
        NARR = self.NBINS * 2 + 1
        start = 250
        end = 5000
        beam = 50

        self.beamheight = np.linspace(start, end, num=NARR)
        self.beamradius = np.ones(NARR) * beam
        self.terrainheight = np.linspace(start - beam, end + beam, num=NARR)
        self.ones = np.ones(int((NARR - 1) / 2))
        self.sample_pbb = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1],
                                    [0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1]])
        self.sample_cbb = np.array([[0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4],
                                    [0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4]])

    def test_beam_block_frac(self):
        """
        terrainheight increases linear through ascending beam with constant
        beamwidth so lower half and reversed upper half elements of pbb array
        add up to one.
        test for equality with ones-array

        """
        pbb = qual.beam_block_frac(self.terrainheight, self.beamheight,
                                   self.beamradius)
        arr = pbb[0:self.NBINS] + pbb[-1:self.NBINS:-1]
        self.assertTrue(np.allclose(arr, self.ones))

    def test_cum_beam_block_frac(self):
        """
        Test whether local maxima BEFORE the absolute maximum along a beam
        are correctly dealt with.

        """
        cbb = qual.cum_beam_block_frac(self.sample_pbb)
        self.assertTrue(np.allclose(cbb, self.sample_cbb))
