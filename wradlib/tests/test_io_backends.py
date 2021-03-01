# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import os

import numpy as np
import pytest
import xarray as xr

from wradlib import io

from . import get_wradlib_data_file, requires_data, requires_xarray_backend_api


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture(params=["v1", "v2"])
def xarray_backend_api(request, monkeypatch):
    monkeypatch.setenv("XARRAY_BACKEND_API", request.param)
    return request.param


@requires_data
@requires_xarray_backend_api
def test_radolan_backend(file_or_filelike, xarray_backend_api):
    filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
    test_attrs = {
        "radarlocations": [
            "boo",
            "ros",
            "emd",
            "hnr",
            "pro",
            "ess",
            "asd",
            "neu",
            "nhb",
            "oft",
            "tur",
            "isn",
            "fbg",
            "mem",
        ],
        "radolanversion": "2.13.1",
        "radarid": "10000",
    }
    with get_wradlib_data_file(filename, file_or_filelike) as rwfile:
        if xarray_backend_api == "v1":
            with pytest.raises(ValueError):
                with io.radolan.open_radolan_dataset(rwfile, engine="radolan") as ds:
                    pass
        else:
            data, meta = io.read_radolan_composite(rwfile)
            data[data == -9999.0] = np.nan
            with xr.open_dataset(rwfile, engine="radolan") as ds:
                assert ds["RW"].encoding["dtype"] == np.uint16
                if file_or_filelike == "file":
                    assert ds["RW"].encoding["source"] == os.path.abspath(rwfile)
                else:
                    assert ds["RW"].encoding["source"] == "None"
                assert ds.attrs == test_attrs
                assert ds["RW"].shape == (900, 900)
                np.testing.assert_almost_equal(
                    ds["RW"].values,
                    (data * 3600.0 / meta["intervalseconds"]),
                    decimal=5,
                )
