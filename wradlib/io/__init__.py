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
.. automodule:: wradlib.io.misc
"""
from .misc import (writePolygon2Text, read_safnwc, # noqa
                 write_raster_dataset, open_shape, open_raster)  # noqa
from .hdf import (read_generic_hdf5, read_OPERA_hdf5, read_GAMIC_hdf5,  # noqa
                  to_hdf5, from_hdf5)  # noqa
from .netcdf import read_EDGE_netcdf, read_generic_netcdf  # noqa
from .rainbow import read_Rainbow  # noqa
from .radolan import readDX, read_RADOLAN_composite  # noqa

__all__ = [s for s in dir() if not s.startswith('_')]
