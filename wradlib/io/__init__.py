#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
Raw Data I/O
============
Please have a look at the tutorial :ref:`/notebooks/fileio/wradlib_radar_formats.ipynb`
for an introduction on how to deal with different file formats.

.. toctree::
    :maxdepth: 2

.. automodule:: wradlib.io.backends
.. automodule:: wradlib.io.dem
.. automodule:: wradlib.io.furuno
.. automodule:: wradlib.io.gdal
.. automodule:: wradlib.io.hdf
.. automodule:: wradlib.io.iris
.. automodule:: wradlib.io.misc
.. automodule:: wradlib.io.netcdf
.. automodule:: wradlib.io.radolan
.. automodule:: wradlib.io.rainbow
.. automodule:: wradlib.io.xarray
.. automodule:: wradlib.io.xarray_depr
"""

from .backends import *
from .dem import *
from .furuno import *
from .gdal import *
from .hdf import *
from .iris import *
from .misc import *
from .netcdf import *
from .radolan import *
from .rainbow import *
from .xarray import *
from .xarray_depr import *

__all__ = [s for s in dir() if not s.startswith("_")]
