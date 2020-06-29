#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np

from wradlib import classify, io, util

from . import has_data, requires_data


@requires_data
class TestHydrometeorClassification:
    if has_data:
        filename = util.get_wradlib_data_file("misc/msf_xband.gz")
        msf = io.get_membership_functions(filename)
        msf_idp = msf[0, 0, :, 0]
        msf_obs = msf[..., 1:]

        hmca = np.array(
            [
                [4.34960938, 15.68457031, 14.62988281],
                [7.78125, 5.49902344, 5.03808594],
                [0.49659729, 0.22286987, 0.86561584],
                [-9.11071777, -1.60217285, 11.15356445],
                [25.6, 25.6, 25.6],
            ]
        )

        msf_val = classify.msf_index_indep(msf_obs, msf_idp, hmca[0])
        fu = classify.fuzzyfi(msf_val, hmca)
        w = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
        prob = classify.probability(fu, w)

    def test_msf_index_indep(self):
        tst = np.array([-20, 10, 110])
        res = np.array(
            [[[[0.0, 0.0, 0.0, 0.0], [5.0, 10.0, 35.0, 40.0], [0.0, 0.0, 0.0, 0.0]]]]
        )
        msf_val = classify.msf_index_indep(self.msf_obs[0:1, 0:1], self.msf_idp, tst)
        np.testing.assert_array_equal(msf_val, res)

    def test_fuzzify(self):
        res = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        fu = classify.fuzzyfi(self.msf_val, self.hmca)
        np.testing.assert_array_equal(fu[0], res)

    def test_probability(self):
        res = np.array(
            [
                [0.16666667, 0.5, 0.66666667],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.21230469, 0.33333333],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.33333333, 0.33333333],
                [0.0, 0.33333333, 0.47532552],
                [0.28997396, 0.5, 0.5],
                [0.28997396, 0.5, 0.33333333],
            ]
        )
        prob = classify.probability(self.fu, self.w)
        np.testing.assert_array_almost_equal(prob, res, decimal=8)

    def test_classify(self):
        res_idx = np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 4, 4],
                [4, 5, 5],
                [5, 6, 6],
                [6, 11, 11],
                [7, 3, 3],
                [8, 7, 7],
                [11, 8, 10],
                [0, 0, 8],
                [9, 9, 9],
                [10, 10, 0],
            ]
        )
        res_vals = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.21230469, 0.33333333],
                [0.0, 0.33333333, 0.33333333],
                [0.0, 0.33333333, 0.33333333],
                [0.16666667, 0.5, 0.47532552],
                [0.28997396, 0.5, 0.5],
                [0.28997396, 0.5, 0.66666667],
            ]
        )

        hmc_idx, hmc_vals = classify.classify(self.prob, threshold=0.0)

        np.testing.assert_array_almost_equal(hmc_idx, res_idx)
        np.testing.assert_array_almost_equal(hmc_vals, res_vals)
