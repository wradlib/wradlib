"""
Raw Data I/O
^^^^^^^^^^^^

Please have a look at the tutorial
:ref:`notebooks/fileio/wradlib_radar_formats.ipynb`
for an introduction on how to deal with different file formats.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   writePolygon2Text
   read_EDGE_netcdf
   read_generic_hdf5
   read_generic_netcdf
   read_OPERA_hdf5
   read_GAMIC_hdf5
   read_Rainbow
   read_safnwc
   write_raster_dataset
   to_AAIGrid
   to_GeoTIFF
   to_hdf5
   from_hdf5
   read_raster_data
   open_shape


Read RADOLAN and DX
^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/

    readDX
    read_RADOLAN_composite


"""


from .io import *
from .radolan import *