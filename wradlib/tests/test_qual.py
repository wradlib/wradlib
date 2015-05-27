# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_qual.py
# Purpose:     unit tests for the wrdalib.qual module
#
# Author:      Kai Muehlbauer
#
# Created:     26.05.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import numpy as np
import wradlib.qual as qual
import unittest

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
        self.terrainheight = np.linspace(start - beam , end + beam, num=NARR)
        self.ones = np.ones((NARR-1)/2)

    def test_beam_block_frac(self):
        """
        terrainheight increases linear through ascending beam with constant beamwidth
        so lower half and reversed upper half elements of pbb array add up to one.
        test for equality with ones-array

        """
        pbb = qual.beam_block_frac(self.terrainheight, self.beamheight, self.beamradius)
        arr = pbb[0:self.NBINS] + pbb[-1:self.NBINS:-1]
        self.assertTrue(np.allclose(arr, self.ones))





