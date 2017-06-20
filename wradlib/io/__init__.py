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

.. automodule:: wradlib.io.radolan
.. automodule:: wradlib.io.hdf
.. automodule:: wradlib.io.netcdf
.. automodule:: wradlib.io.rainbow
.. automodule:: wradlib.io.gdal
.. automodule:: wradlib.io.misc
"""

from .misc import (writePolygon2Text,  to_pickle, from_pickle)  # noqa
from .hdf import (read_generic_hdf5, read_OPERA_hdf5, read_GAMIC_hdf5,  # noqa
                  to_hdf5, from_hdf5)  # noqa
from .netcdf import read_EDGE_netcdf, read_generic_netcdf  # noqa
from .rainbow import read_Rainbow  # noqa
from .radolan import (readDX, read_RADOLAN_composite,  # noqa
                      read_radolan_header,  get_radolan_filehandle,  # noqa
                      parse_DWD_quant_composite_header)  # noqa
from .gdal import (read_safnwc, write_raster_dataset,  # noqa
                   open_shape, open_raster)  # noqa
__all__ = [s for s in dir() if not s.startswith('_')]
