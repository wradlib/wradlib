#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
wradlib_tests
=============

"""
import contextlib
import importlib.util
import io
import os

import pytest
from packaging.version import Version
from wradlib import util
from xarray import __version__ as xr_version


wradlib_data = util.import_optional("wradlib_data", dep="devel")
has_pooch_data = util.has_import(wradlib_data)
has_env_data = os.environ.get("WRADLIB_DATA", False)

requires_data_folder = pytest.mark.skipif(
    not (has_pooch_data or has_env_data),
    reason="requires wradlib-data package to be installed or 'WRADLIB_DATA' "
    "environment variable set to writable folder.",
)

has_secrets = os.environ.get("WRADLIB_EARTHDATA_BEARER_TOKEN", False)
requires_secrets = pytest.mark.skipif(
    not has_secrets,
    reason="requires 'WRADLIB_EARTHDATA_BEARER_TOKEN' environment variable set.",
)

requires_xarray_backend_api = pytest.mark.skipif(
    (Version(xr_version) < Version("0.17.0")),
    reason="requires xarray version 0.18.0",
)


@contextlib.contextmanager
def get_wradlib_data_file_or_filelike(file, file_or_filelike):
    has_pooch_data = util.has_import(wradlib_data)
    if not has_pooch_data:
        pytest.skip(
            "'wradlib-data' package missing. "
            "Please see 'wradlib-data package' for more information."
        )
    datafile = wradlib_data.DATASETS.fetch(file)
    if file_or_filelike == "filelike":
        _open = open
        if datafile[-3:] == ".gz":
            gzip = util.import_optional("gzip")
            _open = gzip.open
        with _open(datafile, mode="r+b") as f:
            yield io.BytesIO(f.read())
    else:
        yield datafile


def get_wradlib_data_file(file):
    has_pooch_data = util.has_import(wradlib_data)
    if not has_pooch_data:
        pytest.skip(
            "'wradlib-data' package missing. "
            "Please see 'wradlib-data package' for more information."
        )
    return wradlib_data.DATASETS.fetch(file)


bottleneck = util.import_optional("bottleneck")
cartopy = util.import_optional("cartopy")
dask = util.import_optional("dask")
gdal = util.import_optional("osgeo.gdal")
h5py = util.import_optional("h5py")
h5netcdf = util.import_optional("h5netcdf")
mpl = util.import_optional("matplotlib")
netCDF4 = util.import_optional("netCDF4")
ogr = util.import_optional("osgeo.ogr")
osr = util.import_optional("osgeo.osr")
rioxarray = util.import_optional("rioxarray")
pyproj = util.import_optional("pyproj")
requests = util.import_optional("requests")
xmltodict = util.import_optional("xmltodict")


requires_bottleneck = pytest.mark.skipif(
    not util.has_import(bottleneck),
    reason="requires bottleneck.",
)

requires_dask = pytest.mark.skipif(
    not util.has_import(dask),
    reason="requires dask.",
)

requires_xmltodict = pytest.mark.skipif(
    not util.has_import(xmltodict),
    reason="requires xmltodict.",
)

requires_requests = pytest.mark.skipif(
    not util.has_import(requests),
    reason="requires requests.",
)

requires_matplotlib = pytest.mark.skipif(
    not util.has_import(mpl),
    reason="requires matplotlib.",
)

requires_cartopy = pytest.mark.skipif(
    not util.has_import(cartopy),
    reason="requires cartopy.",
)

requires_netcdf = pytest.mark.skipif(
    not util.has_import(netCDF4),
    reason="requires netCDF4.",
)

requires_h5netcdf = pytest.mark.skipif(
    not util.has_import(h5netcdf),
    reason="requires h5netcdf.",
)

requires_h5py = pytest.mark.skipif(
    not util.has_import(h5py),
    reason="requires h5py.",
)

requires_gdal = pytest.mark.skipif(
    not util.has_import(gdal),
    reason="requires gdal.",
)

if not util.has_import(gdal):
    has_geos = False
else:
    has_geos = util.has_geos()

requires_geos = pytest.mark.skipif(
    not has_geos, reason="GDAL missing, or GDAL without GEOS"
)
