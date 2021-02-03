#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np

from wradlib import qual


class TestHelperFunctions:
    def test_get_bb_ratio(self):
        heights = np.array(
            [[1100, 1100], [1100, 1100], [1100, 1100], [1100, 1100]], dtype=np.float_
        )
        widths = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float_)
        quality = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=np.float_)
        z = np.array(
            [
                [[1075, 1100, 1125, 900], [1000, 1090, 1110, 1300]],
                [[1075, 1100, 1125, 900], [1000, 1090, 1110, 1300]],
                [[1075, 1100, 1125, 900], [1000, 1090, 1110, 1300]],
                [[1075, 1100, 1125, 900], [1000, 1090, 1110, 1300]],
            ],
            dtype=np.float_,
        )
        ratio_out = np.array(
            [
                [[0.0, 0.5, 1.0, -3.5], [-1.5, 0.3, 0.7, 4.5]],
                [[0.0, 0.5, 1.0, -3.5], [-1.5, 0.3, 0.7, 4.5]],
                [[0.0, 0.5, 1.0, -3.5], [-1.5, 0.3, 0.7, 4.5]],
                [[0.0, 0.5, 1.0, -3.5], [-1.5, 0.3, 0.7, 4.5]],
            ]
        )
        index_out = np.array([[True, True], [True, True], [True, False], [False, True]])
        ratio, index = qual.get_bb_ratio(heights, widths, quality, z)
        np.testing.assert_array_equal(ratio, ratio_out)
        np.testing.assert_array_equal(index, index_out)

    def test_pulse_volume(self):
        vol_out = np.array(
            [
                0.00000000e00,
                2.39258109e06,
                9.57032436e06,
                2.15332298e07,
                3.82812974e07,
                5.98145272e07,
                8.61329192e07,
                1.17236473e08,
                1.53125190e08,
                1.93799068e08,
            ]
        )
        np.testing.assert_allclose(
            qual.pulse_volume(np.arange(0.0, 100000.0, 10000.0), 100.0, 1.0), vol_out
        )


class TestBeamBlockFrac:
    NBINS = 50
    NARR = NBINS * 2 + 1
    start = 250
    end = 5000
    beam = 50

    beamheight = np.linspace(start, end, num=NARR)
    beamradius = np.ones(NARR) * beam
    terrainheight = np.linspace(start - beam, end + beam, num=NARR)
    ones = np.ones(int((NARR - 1) / 2))
    sample_pbb = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1], [0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1]]
    )
    sample_cbb = np.array(
        [[0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4], [0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4]]
    )

    def test_beam_block_frac(self):
        """
        terrainheight increases linear through ascending beam with constant
        beamwidth so lower half and reversed upper half elements of pbb array
        add up to one.
        test for equality with ones-array

        """
        pbb = qual.beam_block_frac(self.terrainheight, self.beamheight, self.beamradius)
        arr = pbb[0 : self.NBINS] + pbb[-1 : self.NBINS : -1]
        assert np.allclose(arr, self.ones)

    def test_cum_beam_block_frac(self):
        """
        Test whether local maxima BEFORE the absolute maximum along a beam
        are correctly dealt with.

        """
        cbb = qual.cum_beam_block_frac(self.sample_pbb)
        assert np.allclose(cbb, self.sample_cbb)
