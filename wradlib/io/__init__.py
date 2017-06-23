#!/usr/bin/env python
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raw Data I/O
============

Please have a look at the tutorial
:ref:`notebooks/fileio/wradlib_radar_formats.ipynb`
for an introduction on how to deal with different file formats.

.. toctree::
    :maxdepth: 2

.. automodule:: wradlib.io.io
"""

from .io import *  # noqa

__all__ = [s for s in dir() if not s.startswith('_')]
