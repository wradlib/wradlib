# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
wradlib_tests
=============

"""
import os

import pytest

has_data = os.environ.get("WRADLIB_DATA", False)
requires_data = pytest.mark.skipif(
    not has_data,
    reason="requires 'WRADLIB_DATA' environment variable set to wradlib-data repository location.",
)
