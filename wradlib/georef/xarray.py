#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   georeference_dataset
"""

import numpy as np
from . import polar


def georeference_dataset(ds):
    """Georeference Dataset.

    This function adds georeference data to xarray dataset `ds`.

    Parameters
    ----------
    ds : xarray dataset

    Returns
    ----------
    ds : xarray dataset
    """
    # adding xyz aeqd-coordinates
    site = (ds.coords['longitude'].values, ds.coords['latitude'].values,
            ds.coords['altitude'].values)
    dim0 = ds['azimuth'].dims[0]
    xyz, aeqd = polar.spherical_to_xyz(ds['range'],
                                       ds['azimuth'],
                                       ds['elevation'],
                                       site,
                                       squeeze=True)
    gr = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
    ds.coords['x'] = ([dim0, 'range'], xyz[..., 0])
    ds.coords['y'] = ([dim0, 'range'], xyz[..., 1])
    ds.coords['z'] = ([dim0, 'range'], xyz[..., 2])
    ds.coords['gr'] = ([dim0, 'range'], gr)

    # adding rays, bins coordinates
    if ds.sweep_mode == 'azimuth_surveillance':
        bins, rays = np.meshgrid(ds['range'],
                                 ds['azimuth'],
                                 indexing='xy')
    else:
        bins, rays = np.meshgrid(ds['range'],
                                 ds['elevation'],
                                 indexing='xy')
    ds.coords['rays'] = ([dim0, 'range'], rays)
    ds.coords['bins'] = ([dim0, 'range'], bins)

    return ds
