#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["as_xarray_dataarray", "create_xarray_dataarray"]
__doc__ = __doc__.format("\n   ".join(__all__))

import collections

import numpy as np
import xarray as xr

from wradlib.util import import_optional

osr = import_optional("osgeo.osr")


def as_xarray_dataarray(data, dims, coords):
    """Create Xarray DataArray from NumPy Array

        .. versionadded:: 1.3

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        data array
    dims : dict
        dictionary describing xarray dimensions
    coords : dict
        dictionary describing xarray coordinates

    Returns
    -------
    dataset : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    da = xr.DataArray(data, coords=dims.values(), dims=dims.keys())
    da = da.assign_coords(**coords)
    return da


def create_xarray_dataarray(
    data,
    *,
    r=None,
    phi=None,
    theta=None,
    site=None,
    sweep_mode="azimuth_surveillance",
    rf=1.0,
    **kwargs,
):
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
    site : tuple
        Coordinates of the radar site.
    sweep_mode : str
        Defaults to 'azimuth_surveillance'.
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
    dataset : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    # check coordinate tuple
    if site and len(site) < 3:
        raise ValueError(
            "`site` need to be a sequence of coordinates "
            "`longitude`, `latitude`, `altitude`."
        )

    if phi is None:
        if sweep_mode == "azimuth_surveillance":
            phi = np.arange(data.shape[0], dtype=np.float_)
            phi += (phi[1] - phi[0]) / 2.0
        else:
            phi = 0.0

    if r is None:
        r = np.arange(data.shape[1], dtype=np.float_)
        r += (r[1] - r[0]) / 2.0

    if theta is None:
        if sweep_mode == "rhi":
            theta = np.arange(data.shape[0], dtype=np.float_)
            theta += (theta[1] - theta[0]) / 2.0
        else:
            theta = 0.0

    if np.isscalar(theta):
        theta = np.ones_like(phi) * theta

    if np.isscalar(phi):
        phi = np.ones_like(theta) * phi

    r = r.copy()
    phi = phi.copy()
    theta = theta.copy()

    if site is None:
        site = (0.0, 0.0, 0.0)

    dims = collections.OrderedDict()
    dim0 = kwargs.pop("dim0", "azimuth")
    dim1 = kwargs.pop("dim1", "range")
    ang = theta if sweep_mode == "rhi" else phi
    dims[dim0] = np.arange(ang.shape[0])
    dims[dim1] = r / rf
    coords = {
        "azimuth": ([dim0], phi),
        "elevation": ([dim0], theta),
        "longitude": (site[0]),
        "latitude": (site[1]),
        "altitude": (site[2]),
        "sweep_mode": sweep_mode,
    }

    # create xarray dataarray
    da = as_xarray_dataarray(data, dims=dims, coords=coords)

    return da
