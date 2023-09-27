#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["bin_altitude", "bin_distance", "site_distance", "GeorefMiscMethods"]
__doc__ = __doc__.format("\n   ".join(__all__))

from functools import singledispatch

import numpy as np
from xarray import DataArray, Dataset, apply_ufunc
from xradar.model import get_altitude_attrs, get_range_attrs

from wradlib import util


@singledispatch
def bin_altitude(r, theta, sitealt, *, re=6371000, ke=4.0 / 3.0):
    """Calculates the height of a radar bin taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the bin altitude is calculated as

    .. math::

        h = \\sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \\sin\\theta} - k_e r_e

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray`
        Array broadcastable to the shape of r elevation angles in degrees with 0°
        at horizontal and +90° pointing vertically upwards from the radar
    sitealt : float
        Altitude in [m] a.s.l. of the referencing radar site
    re : float, optional
        earth's radius [m], defaults to 6371000.
    ke : float, optional
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    altitude : :class:`numpy:numpy.ndarray`
        Array of heights of the radar bins in [m]

    """
    reff = ke * re
    sr = reff + sitealt
    return np.sqrt(r**2 + sr**2 + 2 * r * sr * np.sin(np.radians(theta))) - reff


def _apply_ufunc_wrapper(obj, func, **kwargs):
    dim0 = obj.wrl.util.dim0()
    out = apply_ufunc(
        func,
        obj.range.expand_dims(dim={dim0: len(obj[dim0])}).assign_coords(
            {dim0: obj[dim0]}
        ),
        obj.elevation.expand_dims(dim={"range": len(obj.range)}, axis=-1).assign_coords(
            range=obj.range
        ),
        obj.altitude.values,
        input_core_dims=[[dim0, "range"], [dim0, "range"], [None]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )

    return out


@bin_altitude.register(Dataset)
@bin_altitude.register(DataArray)
def _bin_altitude_xarray(obj, **kwargs):
    """Calculates the height of a radar bin taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the bin altitude is calculated as

    .. math::

        h = \\sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \\sin\\theta} - k_e r_e

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        DataArray

    Returns
    ------
    z : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    out = _apply_ufunc_wrapper(obj, bin_altitude, **kwargs)
    out.attrs = get_altitude_attrs()
    out.name = "bin_altitude"
    return out


@singledispatch
def bin_distance(r, theta, sitealt, *, re=6371000, ke=4.0 / 3.0):
    """Calculates great circle distance from radar site to radar bin over \
    spherical earth, taking the refractivity of the atmosphere into account.

    .. math::

        s = k_e r_e \\arctan\\left(
        \\frac{r \\cos\\theta}{r \\cos\\theta + k_e r_e + h}\\right)

    where :math:`h` would be the radar site altitude amsl.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray`
        Array broadcastable to the shape of r elevation angles in degrees with 0°
        at horizontal and +90° pointing vertically upwards from the radar
    sitealt : float
        site altitude [m] amsl.
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : :class:`numpy:numpy.ndarray`
        Array of great circle arc distances [m]
    """
    reff = ke * re
    sr = reff + sitealt
    theta = np.radians(theta)
    return reff * np.arctan(r * np.cos(theta) / (r * np.sin(theta) + sr))


@bin_distance.register(Dataset)
@bin_distance.register(DataArray)
def _bin_distance_xarray(obj, **kwargs):
    """Calculates great circle distance from radar site to radar bin over \
    spherical earth, taking the refractivity of the atmosphere into account.

    .. math::

        s = k_e r_e \\arctan\\left(
        \\frac{r \\cos\\theta}{r \\cos\\theta + k_e r_e + h}\\right)

    where :math:`h` would be the radar site altitude amsl.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        DataArray or Dataset

    Returns
    ------
    bin_distance : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    out = _apply_ufunc_wrapper(obj, bin_altitude)
    out.attrs = get_range_attrs()
    out.name = "bin_distance"
    return out


@singledispatch
def site_distance(r, theta, binalt, *, re=6371000, ke=4.0 / 3.0):
    """Calculates great circle distance from bin at certain altitude to the \
    radar site over spherical earth, taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the site distance may be calculated as

    .. math::

        s = k_e r_e \\arcsin\\left(
        \\frac{r \\cos\\theta}{k_e r_e + h_n(r, \\theta, r_e, k_e)}\\right)

    where :math:`h_n` would be provided by
    :func:`~wradlib.georef.misc.bin_altitude`.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray`
        Array broadcastable to the shape of r elevation angles in degrees with 0°
        at horizontal and +90° pointing vertically upwards from the radar
    binalt : :class:`numpy:numpy.ndarray`
        site altitude [m] amsl. same shape as r.
    re : float, optional
        earth's radius [m], defaults to 6371000.
    ke : float, optional
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : :class:`numpy:numpy.ndarray`
        Array of great circle arc distances [m]
    """
    reff = ke * re
    return reff * np.arcsin(r * np.cos(np.radians(theta)) / (reff + binalt))


@site_distance.register(Dataset)
@site_distance.register(DataArray)
def _site_distance_xarray(obj, **kwargs):
    """Calculates great circle distance from bin at certain altitude to the \
    radar site over spherical earth, taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the site distance may be calculated as

    .. math::

        s = k_e r_e \\arcsin\\left(
        \\frac{r \\cos\\theta}{k_e r_e + h_n(r, \\theta, r_e, k_e)}\\right)

    where :math:`h_n` would be provided by
    :func:`~wradlib.georef.misc.bin_altitude`.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        DataArray or Dataset

    Returns
    ------
    z : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    binalt = bin_altitude(obj)
    out = apply_ufunc(
        site_distance,
        binalt.range.expand_dims(dim={dim0: len(binalt.azimuth)}).assign_coords(
            {dim0: binalt[dim0]}
        ),
        binalt.elevation.expand_dims(
            dim={"range": len(binalt.range)}, axis=-1
        ).assign_coords(range=binalt.range),
        binalt,
        input_core_dims=[
            [dim0, "range"],
            [dim0, "range"],
            [dim0, "range"],
        ],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.attrs = get_range_attrs()
    out.name = "site_distance"
    return out


class GeorefMiscMethods:
    """wradlib xarray SubAccessor methods for Georef Misc Methods."""

    @util.docstring(_bin_altitude_xarray)
    def bin_altitude(self, *args, **kwargs):
        if not isinstance(self, GeorefMiscMethods):
            return bin_altitude(self, *args, **kwargs)
        else:
            return bin_altitude(self._obj, *args, **kwargs)

    @util.docstring(_bin_distance_xarray)
    def bin_distance(self, *args, **kwargs):
        if not isinstance(self, GeorefMiscMethods):
            return bin_distance(self, *args, **kwargs)
        else:
            return bin_distance(self._obj, *args, **kwargs)

    @util.docstring(_site_distance_xarray)
    def site_distance(self, *args, **kwargs):
        if not isinstance(self, GeorefMiscMethods):
            return site_distance(self, *args, **kwargs)
        else:
            return site_distance(self._obj, *args, **kwargs)
