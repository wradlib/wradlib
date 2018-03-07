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
"""

from .misc import (bin_altitude, bin_distance, site_distance, get_earth_radius)

from .polar import (centroid_to_polyvert, spherical_to_xyz, spherical_to_proj,
                    spherical_to_polyvert, spherical_to_centroids,
                    sweep_centroids)

from .rect import (get_radolan_coords, get_radolan_grid, xyz_to_spherical)

from .projection import (create_osr, proj4_to_osr, reproject,
                         get_default_projection, epsg_to_osr,
                         wkt_to_osr)

from .raster import (pixel_coordinates, pixel_to_map, pixel_to_map3d,
                     read_gdal_coordinates, read_gdal_values,
                     read_gdal_projection, create_raster_dataset,
                     set_raster_origin, extract_raster_dataset,
                     reproject_raster_dataset)

from .satellite import (correct_parallax, dist_from_orbit)

from .vector import (get_vector_points, transform_geometry,
                     get_vector_coordinates, ogr_create_layer, ogr_copy_layer,
                     ogr_copy_layer_by_name, ogr_add_feature, ogr_add_geometry,
                     numpy_to_ogr, ogr_to_numpy, ogr_geocol_to_numpy,
                     get_centroid)

__all__ = [s for s in dir() if not s.startswith('_')]
