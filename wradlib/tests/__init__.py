# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
wradlib_tests
=============

"""
import contextlib
import io
import os

import pytest
from packaging.version import Version
from xarray import __version__ as xr_version

from wradlib import util

has_data = os.environ.get("WRADLIB_DATA", False)
requires_data = pytest.mark.skipif(
    not has_data,
    reason="requires 'WRADLIB_DATA' environment variable set to wradlib-data repository location.",
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
def get_wradlib_data_file(file, file_or_filelike):
    datafile = util.get_wradlib_data_file(file)
    if file_or_filelike == "filelike":
        _open = open
        if datafile[-3:] == ".gz":
            gzip = util.import_optional("gzip")
            _open = gzip.open
        with _open(datafile, mode="r+b") as f:
            yield io.BytesIO(f.read())
    else:
        yield datafile


dask = util.import_optional("dask")
netCDF4 = util.import_optional("netCDF4")
h5py = util.import_optional("h5py")
h5netcdf = util.import_optional("h5netcdf")
gdal = util.import_optional("osgeo.gdal")
ogr = util.import_optional("osgeo.ogr")
osr = util.import_optional("osgeo.osr")
mpl = util.import_optional("matplotlib")
cartopy = util.import_optional("cartopy")
requests = util.import_optional("requests")
xmltodict = util.import_optional("xmltodict")


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
