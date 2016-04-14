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
        self.ones = np.ones((NARR - 1) / 2)

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
