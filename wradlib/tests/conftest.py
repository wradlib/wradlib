#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import pytest


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture
def mock_wradlib_data_env(monkeypatch, tmpdir):
    wradlib_path = tmpdir.mkdir("wradlib-data")
    monkeypatch.setenv("WRADLIB_DATA", wradlib_path)
