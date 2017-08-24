#!/usr/bin/env python
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa

"""
Raw Data I/O
============
Please have a look at the tutorial :ref:`notebooks/fileio/wradlib_radar_formats.ipynb`
for an introduction on how to deal with different file formats.

.. toctree::
    :maxdepth: 2

.. automodule:: wradlib.io.radolan
.. automodule:: wradlib.io.hdf
.. automodule:: wradlib.io.netcdf
.. automodule:: wradlib.io.rainbow
.. automodule:: wradlib.io.gdal
.. automodule:: wradlib.io.iris
.. automodule:: wradlib.io.misc
"""

from .misc import (writePolygon2Text,  to_pickle, from_pickle)
from .hdf import (read_generic_hdf5, read_OPERA_hdf5, read_GAMIC_hdf5,
                  to_hdf5, from_hdf5)
from .netcdf import read_EDGE_netcdf, read_generic_netcdf
from .rainbow import read_Rainbow
from .radolan import (readDX, read_RADOLAN_composite,
                      read_radolan_header,  get_radolan_filehandle,
                      parse_DWD_quant_composite_header,
                      read_radolan_binary_array,
                      decode_radolan_runlength_array)
from .gdal import (read_safnwc, write_raster_dataset,
                   open_shape, open_raster, gdal_create_dataset)
from .iris import (IrisFile, read_iris)

__all__ = [s for s in dir() if not s.startswith('_')]
