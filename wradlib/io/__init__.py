#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# flake8: noqa

"""
Raw Data I/O
============
Please have a look at the tutorial :ref:`/notebooks/fileio/wradlib_radar_formats.ipynb`
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

from .misc import (write_polygon_to_text, to_pickle, from_pickle)
from .gdal import (read_safnwc, write_raster_dataset, open_vector, open_raster,
                   gdal_create_dataset)
from .hdf import (read_generic_hdf5, read_opera_hdf5, read_gamic_hdf5,
                  to_hdf5, from_hdf5, read_gpm, read_trmm)
from .netcdf import read_edge_netcdf, read_generic_netcdf
from .rainbow import read_rainbow
from .radolan import (read_dx, read_radolan_composite,
                      read_radolan_header, get_radolan_filehandle,
                      parse_dwd_composite_header,
                      read_radolan_binary_array,
                      decode_radolan_runlength_array)
from .iris import (IrisFile, read_iris)

__all__ = [s for s in dir() if not s.startswith('_')]
