# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
wradlib_tests
=============

"""

from . import (
    test_adjust,
    test_atten,
    test_classify,
    test_clutter,
    test_comp,
    test_dp,
    test_georef,
    test_io,
    test_io_odim,
    test_ipol,
    test_qual,
    test_trafo,
    test_util,
    test_verify,
    test_vpr,
    test_zonalstats,
    test_zr,
)

__all__ = [s for s in dir() if not s.startswith("_")]
