#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2017, wradlib developers.
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
"""

from .misc import (beam_height_n, arc_distance_n, get_earth_radius,
                   get_shape_coordinates)

from .polar import (polar2lonlat, polar2lonlatalt, polar2lonlatalt_n,
                    centroid2polyvert, polar2polyvert, polar2centroids,
                    projected_bincoords_from_radarspecs, sweep_centroids)

from .rect import (get_radolan_coords, get_radolan_grid)

from .projection import (create_osr, proj4_to_osr, reproject,
                         get_default_projection, epsg_to_osr,
                         wkt_to_osr)

from .raster import (pixel_coordinates, pixel_to_map, pixel_to_map3d,
                     read_gdal_coordinates, read_gdal_values,
                     read_gdal_projection, create_raster_dataset,
                     set_raster_origin, extract_raster_dataset,
                     reproject_raster_dataset)

from .satellite import (correct_parallax, sat2pol, dist_from_orbit)

__all__ = [s for s in dir() if not s.startswith('_')]
