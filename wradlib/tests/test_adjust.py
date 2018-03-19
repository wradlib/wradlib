#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import wradlib.adjust as adjust
import numpy as np

# Arguments to be used throughout all test classes
raw_x, raw_y = np.meshgrid(np.arange(4).astype("f4"),
                           np.arange(4).astype("f4"))
raw_coords = np.vstack((raw_x.ravel(), raw_y.ravel())).T
obs_coords = np.array([[1., 1.], [2., 1.], [1., 3.5], [3.5, 3.]])
raw = np.array([[1., 2., 1., 0., 1., 2., 1.,
                2., 1., 0., 0., 3., 4., 0., 4., 0.],
                [1., 2., 1., 0., 1., 2., 1.,
                 2., 1., 0., 0., 3., 4., 0., 4., 0.]
                ]).T
obs = np.array([[2., 3, 0., 4.], [2., 3, 0., 4.]]).T
nnear_raws = 2
mingages = 3


class AdjustBaseTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = raw_coords
        self.obs_coords = obs_coords
        self.raw = raw
        self.obs = obs
        self.nnear_raws = nnear_raws
        self.mingages = mingages

    def test___init__(self):
        pass

    def test__checkip(self):
        pass

    def test__check_shape(self):
        pass

    def test___call__(self):
        pass

    def test__get_valid_pairs(self):
        pass

    def test_xvalidate(self):
        pass


class AdjustAddTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = raw_coords
        self.obs_coords = obs_coords
        self.raw = raw
        self.obs = obs
        self.nnear_raws = nnear_raws
        self.mingages = mingages

    def test_AdjustAdd_1(self):
        adj = adjust.AdjustAdd(self.obs_coords, self.raw_coords,
                               nnear_raws=self.nnear_raws,
                               mingages=self.mingages)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([[1.62818784, 1.62818784],
                             [2.75926679, 2.75926679],
                             [2.09428144, 2.09428144],
                             [1.1466651, 1.1466651],
                             [1.51948941, 1.51948941],
                             [2.5, 2.5],
                             [2.5, 2.5],
                             [3.27498305, 3.27498305],
                             [1.11382822, 1.11382822],
                             [0.33900645, 0.33900645],
                             [0.89999998, 0.89999998],
                             [4.52409637, 4.52409637],
                             [3.08139533, 3.08139533],
                             [0., 0.],
                             [3.99180328, 3.99180328],
                             [2.16913891, 2.16913891]])
        self.assertTrue(np.allclose(res, shouldbe))
        # test in case only one dataset is passed
        res = adj(self.obs[:, 0], self.raw[:, 0])
        self.assertTrue(np.allclose(res, shouldbe[:, 0]))


class AdjustMultiplyTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = raw_coords
        self.obs_coords = obs_coords
        self.raw = raw
        self.obs = obs
        self.nnear_raws = nnear_raws
        self.mingages = mingages

    def test_AdjustMultiply_1(self):
        adj = adjust.AdjustMultiply(self.obs_coords, self.raw_coords,
                                    nnear_raws=self.nnear_raws,
                                    mingages=self.mingages)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([[1.44937706, 1.44937706],
                             [3.04539442, 3.04539442],
                             [1.74463618, 1.74463618],
                             [0., 0.],
                             [1.37804615, 1.37804615],
                             [2.66666675, 2.66666675],
                             [2., 2.],
                             [3.74106812, 3.74106812],
                             [1.17057478, 1.17057478],
                             [0., 0.],
                             [0., 0.],
                             [6.14457822, 6.14457822],
                             [2.43439031, 2.43439031],
                             [0., 0.],
                             [4.60765028, 4.60765028],
                             [0., 0.]])
        self.assertTrue(np.allclose(res, shouldbe))
        # test in case only one dataset is passed
        res = adj(self.obs[:, 0], self.raw[:, 0])
        self.assertTrue(np.allclose(res, shouldbe[:, 0]))


class AdjustMixedTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = raw_coords
        self.obs_coords = obs_coords
        self.raw = raw
        self.obs = obs
        self.nnear_raws = nnear_raws
        self.mingages = mingages

    def test_AdjustMixed_1(self):
        adj = adjust.AdjustMixed(self.obs_coords, self.raw_coords,
                                 nnear_raws=self.nnear_raws,
                                 mingages=self.mingages)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([[1.51427719, 1.51427719],
                             [2.95735525, 2.95735525],
                             [1.85710269, 1.85710269],
                             [0.36806121, 0.36806121],
                             [1.43181512, 1.43181512],
                             [2.61538471, 2.61538471],
                             [2.15384617, 2.15384617],
                             [3.59765723, 3.59765723],
                             [1.18370627, 1.18370627],
                             [0.15027952, 0.15027952],
                             [0.30825174, 0.30825174],
                             [5.63558862, 5.63558862],
                             [2.49066845, 2.49066845],
                             [-0.29200733, -0.29200733],
                             [4.31646909, 4.31646909],
                             [0.67854041, 0.67854041]])

        self.assertTrue(np.allclose(res, shouldbe))
        # test in case only one dataset is passed
        res = adj(self.obs[:, 0], self.raw[:, 0])
        self.assertTrue(np.allclose(res, shouldbe[:, 0]))


class AdjustMFBTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = np.array([[0., 0.], [1., 1.]])
        self.obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
        self.raw = np.array([2., 2.])
        self.obs = np.array([4., 4.])
        self.nnear_raws = nnear_raws
        self.mingages = 0
        self.mfb_args = dict(method="mean")

    def test_AdjustMFB_1(self):
        adj = adjust.AdjustMFB(self.obs_coords, self.raw_coords,
                               nnear_raws=self.nnear_raws,
                               mingages=self.mingages, mfb_args=self.mfb_args)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([4., 4.])
        self.assertTrue(np.allclose(res, shouldbe))

        adj = adjust.AdjustMFB(self.obs_coords, self.raw_coords,
                               nnear_raws=self.nnear_raws,
                               mingages=self.mingages,
                               mfb_args=dict(method="median"))
        adj(self.obs, self.raw)
        adj = adjust.AdjustMFB(self.obs_coords, self.raw_coords,
                               nnear_raws=self.nnear_raws,
                               mingages=self.mingages,
                               mfb_args=dict(method="linregr", minslope=1.0,
                                             minr='0.7', maxp=0.5))
        adj(self.obs, self.raw)


class AdjustNoneTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = np.array([[0., 0.], [1., 1.]])
        self.obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
        self.raw = np.array([2., 2.])
        self.obs = np.array([4., 4.])
        self.nnear_raws = nnear_raws
        self.mingages = 0
        self.mfb_args = dict(method="mean")

    def test_AdjustNone_1(self):
        adj = adjust.AdjustNone(self.obs_coords, self.raw_coords,
                                nnear_raws=self.nnear_raws,
                                mingages=self.mingages)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([2., 2.])
        self.assertTrue(np.allclose(res, shouldbe))


class GageOnlyTest(unittest.TestCase):
    def setUp(self):
        self.raw_coords = np.array([[0., 0.], [1., 1.]])
        self.obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
        self.raw = np.array([2., 2.])
        self.obs = np.array([4., 4.])
        self.nnear_raws = nnear_raws
        self.mingages = 0
        self.mfb_args = dict(method="mean")

    def test_GageOnly_1(self):
        adj = adjust.GageOnly(self.obs_coords, self.raw_coords,
                              nnear_raws=self.nnear_raws,
                              mingages=self.mingages)
        res = adj(self.obs, self.raw)
        shouldbe = np.array([4., 4.])
        self.assertTrue(np.allclose(res, shouldbe))


class AdjustHelperTest(unittest.TestCase):
    def test__get_neighbours_ix(self):
        pass

    def test__get_statfunc(self):
        adjust._get_statfunc('median')
        adjust._get_statfunc('best')
        self.assertRaises(NameError,
                          lambda: adjust._get_statfunc('wradlib'))

    def test_best(self):
        x = 7.5
        y = np.array([0., 1., 0., 1., 0., 7.7, 8., 8., 8., 8.])
        self.assertEqual(adjust.best(x, y), 7.7)


if __name__ == '__main__':
    unittest.main()
