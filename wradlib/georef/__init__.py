#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
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
.. automodule:: wradlib.georef.vector
.. automodule:: wradlib.georef.rect
.. automodule:: wradlib.georef.satellite
.. automodule:: wradlib.georef.xarray
"""

from .misc import (bin_altitude, bin_distance, site_distance, get_earth_radius)

from .polar import (centroid_to_polyvert, spherical_to_xyz, spherical_to_proj,
                    spherical_to_polyvert, spherical_to_centroids,
                    sweep_centroids)

from .polar import *

from .rect import (get_radolan_coords, get_radolan_grid, xyz_to_spherical)

from .projection import (create_osr, proj4_to_osr, reproject,
                         get_default_projection, epsg_to_osr,
                         wkt_to_osr, get_earth_projection,
                         get_radar_projection, get_extent,
                         geoid_to_ellipsoid, ellipsoid_to_geoid,
                         get_earth_radius2)

from .raster import (read_gdal_coordinates, read_gdal_values,
                     read_gdal_projection, create_raster_dataset,
                     set_raster_origin, extract_raster_dataset,
                     get_raster_extent, get_raster_elevation,
                     reproject_raster_dataset, merge_rasters,
                     raster_to_polyvert)

from .satellite import (correct_parallax, dist_from_orbit)

from .vector import (get_vector_points, transform_geometry,
                     get_vector_coordinates, ogr_create_layer, ogr_copy_layer,
                     ogr_copy_layer_by_name, ogr_add_feature, ogr_add_geometry,
                     numpy_to_ogr, ogr_to_numpy, ogr_geocol_to_numpy,
                     get_centroid)

from .xarray import (as_xarray_dataarray, create_xarray_dataarray,
                     georeference_dataset)

__all__ = [s for s in dir() if not s.startswith('_')]
