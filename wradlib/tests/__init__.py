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
from distutils.version import LooseVersion

import pytest
from xarray import __version__ as xr_version

from wradlib import util

has_data = os.environ.get("WRADLIB_DATA", False)
requires_data = pytest.mark.skipif(
    not has_data,
    reason="requires 'WRADLIB_DATA' environment variable set to wradlib-data repository location.",
)

has_secrets = os.environ.get("WRADLIB_EARTHDATA_USER", False)
requires_secrets = pytest.mark.skipif(
    not has_secrets,
    reason="requires 'WRADLIB_EARTHDATA_USER' and 'WRADLIB_EARTHDATA_PASS' environment variable set.",
)

requires_xarray_backend_api = pytest.mark.skipif(
    (LooseVersion(xr_version) < LooseVersion("0.17.0")),
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
