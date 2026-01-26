#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import pytest
import xarray as xr

from wradlib import georef, io, util

wradlib_data = util.import_optional("wradlib_data", dep="devel")


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture
def mock_wradlib_data_env(monkeypatch, tmpdir):
    wradlib_path = tmpdir.mkdir("wradlib-data")
    monkeypatch.setenv("WRADLIB_DATA", wradlib_path)


@pytest.fixture(scope="session")
def dx_swp():
    fname = wradlib_data.DATASETS.fetch("dx/raa00-dx_10908-0806021655-fbg---bin.gz")
    data, attrs = io.read_dx(fname)
    return georef.create_xarray_dataarray(data)


@pytest.fixture(scope="session")
def dwd_swp():
    fname = wradlib_data.DATASETS.fetch("dx/raa00-dx_10908-0806021655-fbg---bin.gz")
    data, attrs = io.read_dx(fname)
    return georef.create_xarray_dataarray(data)


@pytest.fixture(scope="session")
def gamic_swp():
    fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
    swp = xr.open_dataset(fname, engine="gamic", group="sweep_0")
    return swp


@pytest.fixture(scope="session")
def dem_bonn():
    fname = wradlib_data.DATASETS.fetch("geo/bonn_new.tif")
    return xr.open_dataset(fname, engine="rasterio")
