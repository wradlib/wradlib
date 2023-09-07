#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import xarray

import wradlib


def test_accessor():
    da = xarray.DataArray()

    print(type(da.wrl.vis.plot))
    print(type(wradlib.vis.VisMethods.plot))
