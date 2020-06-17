#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa
"""
Georeferencing
^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

.. automodule:: wradlib.georef.misc
.. automodule:: wradlib.georef.polar
.. automodule:: wradlib.georef.projection
.. automodule:: wradlib.georef.raster
.. automodule:: wradlib.georef.rect
.. automodule:: wradlib.georef.satellite
.. automodule:: wradlib.georef.vector
.. automodule:: wradlib.georef.xarray
"""

from .misc import *
from .polar import *
from .projection import *
from .raster import *
from .rect import *
from .satellite import *
from .vector import *
from .xarray import *

__all__ = [s for s in dir() if not s.startswith("_")]
