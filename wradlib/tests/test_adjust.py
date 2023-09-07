#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import adjust


@pytest.fixture
def adjust_data():
    @dataclass(init=False, repr=False, eq=False)
    class DataClass:
        # Arguments to be used throughout all test classes
        raw_x, raw_y = np.meshgrid(np.arange(4).astype("f4"), np.arange(4).astype("f4"))
        raw_coords = np.vstack((raw_x.ravel(), raw_y.ravel())).T
        obs_coords = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 3.5], [3.5, 3.0]])
        raw = np.array(
            [
                [
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    3.0,
                    4.0,
                    0.0,
                    4.0,
                    0.0,
                ],
                [
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    3.0,
                    4.0,
                    0.0,
                    4.0,
                    0.0,
                ],
            ]
        ).T
        obs = np.array([[2.0, 3, 0.0, 4.0], [2.0, 3, 0.0, 4.0]]).T
        nnear_raws = 2
        mingages = 3

    yield DataClass


def test_AdjustAdd_1(adjust_data):
    adj = adjust.AdjustAdd(
        adjust_data.obs_coords,
        adjust_data.raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=adjust_data.mingages,
    )
    res = adj(adjust_data.obs, adjust_data.raw)
    shouldbe = np.array(
        [
            [1.62818784, 1.62818784],
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
            [0.0, 0.0],
            [3.99180328, 3.99180328],
            [2.16913891, 2.16913891],
        ]
    )
    assert np.allclose(res, shouldbe)
    # test in case only one dataset is passed
    res = adj(adjust_data.obs[:, 0], adjust_data.raw[:, 0])
    assert np.allclose(res, shouldbe[:, 0])


def test_AdjustMultiply_1(adjust_data):
    adj = adjust.AdjustMultiply(
        adjust_data.obs_coords,
        adjust_data.raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=adjust_data.mingages,
    )
    res = adj(adjust_data.obs, adjust_data.raw)
    shouldbe = np.array(
        [
            [1.44937706, 1.44937706],
            [3.04539442, 3.04539442],
            [1.74463618, 1.74463618],
            [0.0, 0.0],
            [1.37804615, 1.37804615],
            [2.66666675, 2.66666675],
            [2.0, 2.0],
            [3.74106812, 3.74106812],
            [1.17057478, 1.17057478],
            [0.0, 0.0],
            [0.0, 0.0],
            [6.14457822, 6.14457822],
            [2.43439031, 2.43439031],
            [0.0, 0.0],
            [4.60765028, 4.60765028],
            [0.0, 0.0],
        ]
    )
    assert np.allclose(res, shouldbe)
    # test in case only one dataset is passed
    res = adj(adjust_data.obs[:, 0], adjust_data.raw[:, 0])
    assert np.allclose(res, shouldbe[:, 0])


def test_AdjustMixed_1(adjust_data):
    adj = adjust.AdjustMixed(
        adjust_data.obs_coords,
        adjust_data.raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=adjust_data.mingages,
    )
    res = adj(adjust_data.obs, adjust_data.raw)
    shouldbe = np.array(
        [
            [1.51427719, 1.51427719],
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
            [0.67854041, 0.67854041],
        ]
    )

    assert np.allclose(res, shouldbe)
    # test in case only one dataset is passed
    res = adj(adjust_data.obs[:, 0], adjust_data.raw[:, 0])
    assert np.allclose(res, shouldbe[:, 0])


def test_AdjustMFB_1(adjust_data):
    raw_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    raw = np.array([2.0, 2.0])
    obs = np.array([4.0, 4.0])
    mingages = 0
    mfb_args = dict(method="mean")

    adj = adjust.AdjustMFB(
        obs_coords,
        raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=mingages,
        mfb_args=mfb_args,
    )
    res = adj(obs, raw)
    shouldbe = np.array([4.0, 4.0])
    assert np.allclose(res, shouldbe)

    adj = adjust.AdjustMFB(
        obs_coords,
        raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=mingages,
        mfb_args=dict(method="median"),
    )
    adj(obs, raw)
    adj = adjust.AdjustMFB(
        obs_coords,
        raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=mingages,
        mfb_args=dict(method="linregr", minslope=1.0, minr="0.7", maxp=0.5),
    )
    adj(obs, raw)


def test_AdjustNone_1(adjust_data):
    raw_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    raw = np.array([2.0, 2.0])
    obs = np.array([4.0, 4.0])
    mingages = 0
    dict(method="mean")

    adj = adjust.AdjustNone(
        obs_coords,
        raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=mingages,
    )
    res = adj(obs, raw)
    shouldbe = np.array([2.0, 2.0])
    assert np.allclose(res, shouldbe)


def test_GageOnly_1(adjust_data):
    raw_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    obs_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    raw = np.array([2.0, 2.0])
    obs = np.array([4.0, 4.0])
    mingages = 0
    dict(method="mean")

    adj = adjust.GageOnly(
        obs_coords,
        raw_coords,
        nnear_raws=adjust_data.nnear_raws,
        mingages=mingages,
    )
    res = adj(obs, raw)
    shouldbe = np.array([4.0, 4.0])
    assert np.allclose(res, shouldbe)


def test__get_statfunc():
    adjust._get_statfunc("median")
    adjust._get_statfunc("best")
    with pytest.raises(NameError):
        adjust._get_statfunc("wradlib")


def test_best():
    x = 7.5
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 7.7, 8.0, 8.0, 8.0, 8.0])
    assert adjust.best(x, y) == 7.7
