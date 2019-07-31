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

   as_xarray_dataarray
   create_xarray_dataarray
   georeference_dataset
"""

import numpy as np
import xarray as xr
import collections
from osgeo import osr
from . import polar


def as_xarray_dataarray(data, dims, coords):
    """Create Xarray DataArray from NumPy Array

        .. versionadded:: 1.3

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        data array
    dims : dictionary
        dictionary describing xarray dimensions
    coords : dictionary
        dictionary describing xarray coordinates

    Returns
    -------
    dataset : xr.DataArray
        DataArray
    """
    da = xr.DataArray(data, coords=dims.values(), dims=dims.keys())
    da = da.assign_coords(**coords)
    return da


def create_xarray_dataarray(data, r=None, phi=None, theta=None,
                            site=None, sweep_mode='azimuth_surveillance',
                            rf=1.0, **kwargs):
    """Create Xarray DataArray from Polar Radar Data

        .. versionadded:: 1.3

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        The data array. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : :class:`numpy:numpy.ndarray`
        The ranges. Units may be chosen arbitrarily, m preferred.
    phi : :class:`numpy:numpy.ndarray`
        The azimuth angles in degrees.
    theta : :class:`numpy:numpy.ndarray`
        The elevation angles in degrees.
    proj : osr object
        Destination Spatial Reference System (Projection).
    site : tuple
        Tuple of coordinates of the radar site.
    sweep_mode : str
        Defaults to 'PPI'.
    rf : float
        factor to scale range, defaults to 1. (no scale)

    Keyword Arguments
    -----------------
    re : float
        effective earth radius
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. Defaults to 4/3.
    dim0 : str
        Name of the first dimension. Defaults to 'azimuth'.
    dim1 : str
        Name of the second dimension. Defaults to 'range'.

    Returns
    -------
    dataset : xr.DataArray
        DataArray
    """
    if (r is None) or (phi is None) or (theta is None):
        raise TypeError("wradlib: function `create_xarray_dataarray` requires "
                        "r, phi and theta keyword-arguments.")

    r = r.copy()
    phi = phi.copy()
    theta = theta.copy()

    if site is None:
        site = (0., 0., 0.)

    dims = collections.OrderedDict()
    dim0 = kwargs.pop('dim0', 'azimuth')
    dim1 = kwargs.pop('dim1', 'range')
    dims[dim0] = np.arange(phi.shape[0])
    dims[dim1] = r / rf
    coords = {'azimuth': ([dim0], phi),
              'elevation': ([dim0], theta),
              'longitude': (site[0]),
              'latitude': (site[1]),
              'altitude': (site[2]),
              'sweep_mode': sweep_mode,
              }

    # create xarray dataarray
    da = as_xarray_dataarray(data, dims=dims, coords=coords)

    return da


def georeference_dataset(ds, **kwargs):
    """Georeference Dataset.

    This function adds georeference data to xarray dataset `ds`.

    Parameters
    ----------
    ds : xarray dataset

    Returns
    ----------
    ds : xarray dataset
    """
    proj = kwargs.pop('proj', 'None')
    re = kwargs.pop('re', None)
    ke = kwargs.pop('ke', 4. / 3.)

    # adding xyz aeqd-coordinates
    site = (ds.coords['longitude'].values, ds.coords['latitude'].values,
            ds.coords['altitude'].values)

    if site == (0., 0., 0.):
        re = 6378137.

    dim0 = ds['azimuth'].dims[0]

    # GDAL OSR, convert to this proj
    if isinstance(proj, osr.SpatialReference):
        xyz = polar.spherical_to_proj(ds['range'],
                                      ds['azimuth'],
                                      ds['elevation'],
                                      site,
                                      proj=proj,
                                      re=re,
                                      ke=ke)
    # other proj, convert to aeqd
    elif proj:
        xyz, dst_proj = polar.spherical_to_xyz(ds['range'],
                                               ds['azimuth'],
                                               ds['elevation'],
                                               site,
                                               re=re,
                                               ke=ke,
                                               squeeze=True)
    # proj, convert to aeqd and add offset
    else:
        xyz, dst_proj = polar.spherical_to_xyz(ds['range'],
                                               ds['azimuth'],
                                               ds['elevation'],
                                               site,
                                               re=re,
                                               ke=ke,
                                               squeeze=True)
        xyz += np.array(site).T

    # calculate center point
    center = np.mean(xyz[:, 0, :], axis=0)

    # calculate ground range
    gr = np.sqrt((xyz[..., 0] - center[0]) ** 2 +
                 (xyz[..., 1] - center[1]) ** 2)

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
