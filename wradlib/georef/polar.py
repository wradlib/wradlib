#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Polar Grid Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "georeference",
    "spherical_to_xyz",
    "spherical_to_proj",
    "spherical_to_polyvert",
    "spherical_to_centroids",
    "centroid_to_polyvert",
    "sweep_centroids",
    "maximum_intensity_projection",
    "GeorefPolarMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))
__doctest_requires__ = {"spherical*": ["osgeo"]}

from functools import singledispatch

import numpy as np
from xarray import DataArray, Dataset, apply_ufunc
from xradar.georeference import add_crs

from wradlib.georef import misc, projection
from wradlib.util import docstring, has_import, import_optional, warn

osr = import_optional("osgeo.osr")
pyproj = import_optional("pyproj")


@singledispatch
def spherical_to_xyz(
    r,
    phi,
    theta,
    site,
    *,
    re=None,
    ke=4.0 / 3.0,
    **kwargs,
):
    """Transforms spherical coordinates (r, phi, theta) to cartesian
    coordinates (x, y, z) centered at site (aeqd).

    It takes the shortening of the great circle
    distance with increasing elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Contains the radial distances in meters.
    phi : :class:`numpy:numpy.ndarray`
        Contains the azimuthal angles in degree.
    theta: :class:`numpy:numpy.ndarray`
        Contains the elevation angles in degree.
    site : sequence
        the lon / lat / alt coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
    re : float, optional
        earth's radius [m], defaults to None (calculating from given latitude).
    ke : float, optional
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Keyword Arguments
    -----------------
    squeeze : bool, optional
        If True, returns squeezed array. Defaults to False.
    strict_dims : bool, optional
        If True, generates output of (theta, phi, r, 3) in any case.
        If False, dimensions with same length are "merged".
        Defaults to False.

    Returns
    -------
    xyz : :class:`numpy:numpy.ndarray`
        Array of shape (..., 3). Contains cartesian coordinates.
    aeqd : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (AEQD-Projection).
    """
    squeeze = kwargs.get("squeeze", False)
    strict_dims = kwargs.get("strict_dims", False)

    centalt = site[2]

    # if no radius is given, get the approximate radius of the WGS84
    # ellipsoid for the site's latitude
    if re is None:
        re = projection.get_earth_radius(site[1])
        # Set up aeqd-projection sitecoord-centered, wgs84 datum and ellipsoid
        # use world azimuthal equidistant projection
        projstr = (
            f"+proj=aeqd +lon_0={site[0]:f} +x_0=0 +y_0=0 "
            f"+lat_0={site[1]:f} +ellps=WGS84 +datum=WGS84 "
            "+units=m +no_defs"
        )
    else:
        # Set up aeqd-projection sitecoord-centered, assuming spherical earth
        # use sphere azimuthal equidistant projection
        projstr = (
            f"+proj=aeqd +lon_0={site[0]:f} +lat_0={site[1]:f} "
            f"+a={re:f} +b={re:f} +units=m +no_defs"
        )

    if has_import(osr):
        aeqd = projection.projstr_to_osr(projstr)
    else:
        aeqd = projstr

    r = np.asanyarray(r)
    theta = np.asanyarray(theta)
    phi = np.asanyarray(phi)

    if r.ndim:
        r = r.reshape((1,) * (3 - r.ndim) + r.shape)

    if phi.ndim:
        phi = phi.reshape((1,) + phi.shape + (1,) * (2 - phi.ndim))

    if not theta.ndim:
        theta = np.broadcast_to(theta, phi.shape)

    dims = 3
    if not strict_dims:
        if phi.ndim and theta.ndim and (theta.shape[0] == phi.shape[1]):
            dims -= 1
        if r.ndim and theta.ndim and (theta.shape[0] == r.shape[2]):
            dims -= 1

    if theta.ndim and phi.ndim:
        theta = theta.reshape(theta.shape + (1,) * (dims - theta.ndim))

    z = misc.bin_altitude(r, theta, centalt, re=re, ke=ke)
    dist = misc.site_distance(r, theta, z, re=re, ke=ke)

    if (not strict_dims) and phi.ndim and r.ndim and (r.shape[2] == phi.shape[1]):
        z = np.squeeze(z)
        dist = np.squeeze(dist)
        phi = np.squeeze(phi)

    x = dist * np.cos(np.radians(90 - phi))
    y = dist * np.sin(np.radians(90 - phi))

    if z.ndim:
        z = np.broadcast_to(z, x.shape)

    xyz = np.stack((x, y, z), axis=-1)

    if xyz.ndim == 1:
        xyz.shape = (1,) * 3 + xyz.shape
    elif xyz.ndim == 2:
        xyz.shape = (xyz.shape[0],) + (1,) * 2 + (xyz.shape[1],)

    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz, aeqd


@spherical_to_xyz.register(Dataset)
@spherical_to_xyz.register(DataArray)
def _spherical_to_xyz_xarray(obj, **kwargs):
    """Transforms spherical coordinates (r, phi, theta) to cartesian
    coordinates (x, y, z) centered at site (aeqd).

    It takes the shortening of the great circle
    distance with increasing elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    re : float
        earth's radius [m], defaults to None (calculating from given latitude)
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    -------
    xyz : :py:class:`xarray:xarray.DataArray`
        Array of shape (..., 3). Contains cartesian coordinates.
    aeqd : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (AEQD-Projection).
    """
    dim0 = obj.wrl.util.dim0()
    r = obj.range.expand_dims(dim={dim0: len(obj[dim0])}).assign_coords(
        {dim0: obj[dim0]}
    )
    phi = obj.azimuth.expand_dims(dim={"range": len(obj.range)}, axis=-1).assign_coords(
        range=obj.range
    )
    theta = obj.elevation
    site = (obj.longitude.values, obj.latitude.values, obj.altitude.values)
    kwargs.setdefault("squeeze", True)
    out, aeqd = apply_ufunc(
        spherical_to_xyz,
        r,
        phi,
        theta,
        site,
        input_core_dims=[
            [dim0, "range"],
            [dim0, "range"],
            [dim0],
            [None],
        ],
        output_core_dims=[[dim0, "range", "xyz"], []],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "spherical_to_xyz"
    return out, aeqd


@singledispatch
def spherical_to_proj(r, phi, theta, site, *, crs=None, re=None, ke=4.0 / 3.0):
    """Transforms spherical coordinates (r, phi, theta) to projected
    coordinates centered at site in given projection.

    It takes the shortening of the great circle
    distance with increasing elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Contains the radial distances.
    phi : :class:`numpy:numpy.ndarray`
        Contains the azimuthal angles.
    theta: :class:`numpy:numpy.ndarray`
        Contains the elevation angles.
    site : sequence
        the lon / lat coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
        if site is of length two, altitude is assumed to be zero
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`, optional
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    re : float, optional
        earth's radius [m], defaults to None (calculating from given latitude).
    ke : float, optional
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 3). Contains projected map coordinates.

    Examples
    --------

    A few standard directions (North, South, North, East, South, West) with
    different distances (amounting to roughly 1°) from a site
    located at 48°N 9°E

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])*1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> th = np.array([0.,   0.,   0.,   0.,   0.,  0.5,])
    >>> csite = (9.0, 48.0, 0)
    >>> coords = spherical_to_proj(r, az, th, csite)
    >>> for coord in coords:
    ...     print( '{0:7.4f}, {1:7.4f}, {2:7.4f}'.format(*coord))
    ...
     9.0000, 48.0000,  0.0000
     9.0000, 48.0000,  0.0000
     9.0000, 48.9981, 725.7160
    10.4872, 47.9904, 725.7160
     9.0000, 47.0017, 725.7160
     7.5131, 47.9904, 1694.2234

    Here, the coordinates of the east and west directions won't come to lie on
    the latitude of the site because the beam doesn't travel along the latitude
    circle but along a great circle.

    See :ref:`/notebooks/basics/wradlib_workflow.ipynb#\
Georeferencing-and-Projection`.
    """
    if crs is None:
        crs = projection.get_default_projection()

    xyz, aeqd = spherical_to_xyz(r, phi, theta, site, re=re, ke=ke, squeeze=True)

    # reproject aeqd to destination projection
    coords = projection.reproject(xyz, src_crs=aeqd, trg_crs=crs)

    return coords


@spherical_to_proj.register(Dataset)
@spherical_to_proj.register(DataArray)
def _spherical_to_proj_xarray(obj, **kwargs):
    """Transforms spherical coordinates (r, phi, theta) to projected
    coordinates centered at site in given projection.

    It takes the shortening of the great circle
    distance with increasing elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    -------
    coords : :py:class:`xarray:xarray.DataArray`
        Array of shape (..., 3). Contains projected map coordinates.
    """
    dim0 = obj.wrl.util.dim0()
    r = obj.range.expand_dims(dim={dim0: len(obj[dim0])}).assign_coords(
        {dim0: obj[dim0]}
    )
    phi = obj.azimuth.expand_dims(dim={"range": len(obj.range)}, axis=-1).assign_coords(
        range=obj.range
    )
    theta = obj.elevation
    site = (obj.longitude.values, obj.latitude.values, obj.altitude.values)
    out = apply_ufunc(
        spherical_to_proj,
        r,
        phi,
        theta,
        site,
        input_core_dims=[
            [dim0, "range"],
            [dim0, "range"],
            [dim0],
            [None],
        ],
        output_core_dims=[[dim0, "range", "xyz"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "spherical_to_proj"
    return out


def centroid_to_polyvert(centroid, delta, /):
    """Calculates the 2-D Polygon vertices necessary to form a rectangular
    polygon around the centroid's coordinates.

    The vertices order will be clockwise, as this is the convention used
    by ESRI's shapefile format for a polygon.

    Parameters
    ----------
    centroid : array-like
        List of 2-D coordinates of the center point of the rectangle.
    delta : scalar or :class:`numpy:numpy.ndarray`
        Symmetric distances of the vertices from the centroid in each
        direction. If ``delta`` is scalar, it is assumed to apply to
        both dimensions.

    Returns
    -------
    vertices : :class:`numpy:numpy.ndarray`
        An array with 5 vertices per centroid.

    Note
    ----
    The function can currently only deal with 2-D data (If you come up with a
    higher dimensional version of 'clockwise' you're welcome to add it).
    The data is then assumed to be organized within the ``centroid`` array with
    the last dimension being the 2-D coordinates of each point.

    Examples
    --------

    >>> centroid_to_polyvert([0., 1.], [0.5, 1.5])
    array([[-0.5, -0.5],
           [-0.5,  2.5],
           [ 0.5,  2.5],
           [ 0.5, -0.5],
           [-0.5, -0.5]])
    >>> centroid_to_polyvert(np.arange(4).reshape((2,2)), 0.5)
    array([[[-0.5,  0.5],
            [-0.5,  1.5],
            [ 0.5,  1.5],
            [ 0.5,  0.5],
            [-0.5,  0.5]],
    <BLANKLINE>
           [[ 1.5,  2.5],
            [ 1.5,  3.5],
            [ 2.5,  3.5],
            [ 2.5,  2.5],
            [ 1.5,  2.5]]])

    """
    cent = np.asanyarray(centroid)
    if cent.shape[-1] != 2:
        raise ValueError("Parameter `centroid` dimensions need to be (..., 2).")
    dshape = [1] * cent.ndim
    dshape.insert(-1, 5)
    dshape[-1] = 2

    d = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    ).reshape(tuple(dshape))

    return np.asanyarray(centroid)[..., None, :] + d * np.asanyarray(delta)


@singledispatch
def spherical_to_polyvert(r, phi, theta, site, *, crs=None):
    """
    Generate 3-D polygon vertices directly from spherical coordinates
    (r, phi, theta).

    This is an alternative to :func:`~wradlib.georef.polar.centroid_to_polyvert`
    which does not use centroids, but generates the polygon vertices by simply
    connecting the corners of the radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    :func:`~wradlib.georef.polar.spherical_to_xyz`.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]; r defines the exterior boundaries of the range
        bins! (not the centroids). Thus, values must be positive!
    phi : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main
        beam lobe!
        The first angle can start at any values, but make sure the array is
        sorted continuously positively clockwise and the angles are
        equidistant. An angle if 0 degree is pointing north.
    theta : float
        Elevation angle of scan
    site : sequence
        the lon/lat/alt coordinates of the radar location
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape(num_vertices,
        num_vertex_nodes, 2). The last dimension carries the xyz-coordinates
        either in `aeqd` or given crs.
    aeqd : :py:class:`gdal:osgeo.aeqosr.SpatialReference`
        only returned if crs is None

    Examples
    --------
    >>> import wradlib.georef as georef  # noqa
    >>> import numpy as np
    >>> from matplotlib import collections
    >>> import matplotlib.pyplot as plt
    >>> # define the polar coordinates and the site coordinates in lat/lon
    >>> r = np.array([50., 100., 150., 200.]) * 1000
    >>> az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
    >>> el = 1.0
    >>> site = (9.0, 48.0, 0)
    >>> polygons, aeqd = georef.spherical_to_polyvert(r, az, el, site)
    >>> # plot the resulting mesh
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> polycoll = collections.PolyCollection(polygons[...,:2], closed=True, facecolors='None')  # noqa
    >>> ret = ax.add_collection(polycoll, autolim=True)
    >>> plt.autoscale()
    >>> plt.show()

    """
    # prepare the range and azimuth array, so they describe the boundaries of
    # a bin, not the centroid
    r, phi = _check_polar_coords(r, phi)
    r = np.insert(r, 0, r[0] - _get_range_resolution(r))
    phi = phi - 0.5 * _get_azimuth_resolution(phi)
    phi = np.append(phi, phi[0])
    phi = np.where(phi < 0, phi + 360.0, phi)

    # generate a grid of polar coordinates of bin corners
    r, phi = np.meshgrid(r, phi)

    coords, aeqd = spherical_to_xyz(r, phi, theta, site, squeeze=True, strict_dims=True)
    if crs is not None:
        coords = projection.reproject(coords, src_crs=aeqd, trg_crs=crs)

    llc = coords[:-1, :-1]
    ulc = coords[:-1, 1:]
    urc = coords[1:, 1:]
    lrc = coords[1:, :-1]

    vertices = np.stack((llc, ulc, urc, lrc, llc), axis=-2).reshape((-1, 5, 3))

    if crs is None:
        return vertices, aeqd
    else:
        return vertices


@spherical_to_polyvert.register(Dataset)
@spherical_to_polyvert.register(DataArray)
def _spherical_to_polyvert_xarray(obj, **kwargs):
    """
    Generate 3-D polygon vertices directly from spherical coordinates
    (r, phi, theta).

    This is an alternative to :func:`~wradlib.georef.polar.centroid_to_polyvert`
    which does not use centroids, but generates the polygon vertices by simply
    connecting the corners of the radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    :func:`~wradlib.georef.polar.spherical_to_xyz`.

    Currently only works for PPI.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    -------
    xyz : :py:class:`xarray:xarray.DataArray`
        Array of shape (..., 3). Contains cartesian coordinates.
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    """
    # Todo: check if this works for elevation too
    obj.wrl.util.dim0()
    rdiff = obj.range.diff("range").median() / 2.0
    r = obj.range + rdiff
    phi = obj.azimuth
    theta = obj.elevation.median("azimuth")
    site = (obj.longitude.values, obj.latitude.values, obj.altitude.values)
    output_core_dims = [["bins", "vert", "xy"]]
    if kwargs.get("crs", None) is None:
        output_core_dims.append([])
    keep_attrs = kwargs.pop("keep_attrs", None)
    out = apply_ufunc(
        spherical_to_polyvert,
        r,
        phi,
        theta.values,
        site,
        input_core_dims=[["range"], ["azimuth"], [None], [None]],
        output_core_dims=output_core_dims,
        dask="parallelized",
        kwargs=kwargs,
        keep_attrs=keep_attrs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    if kwargs.get("crs", None) is None:
        out[0].name = "spherical_to_polyvert"
    else:
        out.name = "spherical_to_polyvert"
    return out


@singledispatch
def spherical_to_centroids(r, phi, theta, site, *, crs=None):
    """
    Generate 3-D centroids of the radar bins from the sperical
    coordinates (r, phi, theta).

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. The ranges are assumed to define the exterior
    boundaries of the range bins (thus they must be positive). The angles are
    assumed to describe the pointing direction fo the main beam lobe.

    For further information refer to the documentation of
    :func:`~wradlib.georef.polar.spherical_to_xyz`.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]; r defines the exterior boundaries of the range
        bins! (not the centroids). Thus, values must be positive!
    phi : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main
        beam lobe!
        The first angle can start at any values, but make sure the array is
        sorted continuously positively clockwise and the angles are
        equidistant. An angle if 0 degree is pointing north.
    theta : float
        Elevation angle of scan
    site : sequence
        the lon/lat/alt coordinates of the radar location
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of bin centroids with shape(num_rays, num_bins, 3).
        The last dimension carries the xyz-coordinates
        either in `aeqd` or given crs.
    aeqd : :py:class:`gdal:osgeo.osr.SpatialReference`
        only returned if crs is None

    Note
    ----
    Azimuth angles of 360 deg are internally converted to 0 deg.

    """
    # make sure the range and azimuth angles have the right properties
    r, phi = _check_polar_coords(r, phi)

    r = r - 0.5 * _get_range_resolution(r)

    # generate a polar grid and convert to lat/lon
    r, phi = np.meshgrid(r, phi)

    coords, aeqd = spherical_to_xyz(r, phi, theta, site, squeeze=True)

    if crs is None:
        return coords, aeqd
    else:
        return projection.reproject(coords, src_crs=aeqd, trg_crs=crs)


@spherical_to_centroids.register(Dataset)
@spherical_to_polyvert.register(DataArray)
def _spherical_to_centroids_xarray(obj, **kwargs):
    """
    Generate 3-D centroids of the radar bins from the sperical
    coordinates (r, phi, theta).

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. The ranges are assumed to define the exterior
    boundaries of the range bins (thus they must be positive). The angles are
    assumed to describe the pointing direction fo the main beam lobe.

    For further information refer to the documentation of
    :func:`~wradlib.georef.polar.spherical_to_xyz`.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Currently only works for PPI.

    Returns
    -------
    xyz : :py:class:`xarray:xarray.DataArray`
        Array of shape (..., 3). Contains cartesian coordinates.
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).

    Note
    ----
    Azimuth angles of 360 deg are internally converted to 0 deg.
    """
    # Todo: check if this works for elevation too
    rdiff = obj.range.diff("range").median() / 2.0
    r = obj.range + rdiff
    phi = obj.azimuth
    theta = obj.elevation.median("azimuth")
    site = (obj.longitude.values, obj.latitude.values, obj.altitude.values)
    output_core_dims = [["azimuth", "range", "xyz"]]
    if kwargs.get("crs", None) is None:
        output_core_dims.append([])
    keep_attrs = kwargs.pop("keep_attrs", None)
    out = apply_ufunc(
        spherical_to_centroids,
        r,
        phi,
        theta.values,
        site,
        input_core_dims=[["range"], ["azimuth"], [None], [None]],
        output_core_dims=output_core_dims,
        dask="parallelized",
        kwargs=kwargs,
        keep_attrs=keep_attrs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    if kwargs.get("crs", None) is None:
        out[0].name = "spherical_to_centroids"
    else:
        out.name = "spherical_to_centroids"
    return out


def _check_polar_coords(r, az, /):
    """
    Contains a lot of checks to make sure the polar coordinates are adequate.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        range gates in any unit
    az : :class:`numpy:numpy.ndarray`
        azimuth angles in degree

    """
    r = np.array(r, "f4")
    az = np.array(az, "f4")
    az[az == 360.0] = 0.0
    if 0.0 in r:
        raise ValueError(
            "Invalid polar coordinates: "
            "0 is not a valid range gate specification "
            "(the centroid of a range gate must be positive)."
        )
    if len(np.unique(r)) != len(r):
        raise ValueError(
            "Invalid polar coordinates: "
            "Range gate specification contains duplicate "
            "entries."
        )
    if len(np.unique(az)) != len(az):
        raise ValueError(
            "Invalid polar coordinates: "
            "Azimuth specification contains duplicate entries."
        )
    if not _is_sorted(r):
        raise ValueError("Invalid polar coordinates: Range array must be sorted.")
    if len(np.unique(r[1:] - r[:-1])) > 1:
        raise ValueError("Invalid polar coordinates: Range gates are not equidistant.")
    if len(np.where(az >= 360.0)[0]) > 0:
        raise ValueError(
            "Invalid polar coordinates: "
            "Azimuth angles must not be greater than "
            "or equal to 360 deg."
        )
    if not _is_sorted(az):
        # it is ok if the azimuth angle array is not sorted, but it has to be
        # 'continuously clockwise', e.g. it could start at 90° and stop at °89
        az_right = az[np.where(np.logical_and(az <= 360, az >= az[0]))[0]]
        az_left = az[np.where(az < az[0])]
        if (not _is_sorted(az_right)) or (not _is_sorted(az_left)):
            raise ValueError(
                "Invalid polar coordinates: Azimuth array is not sorted clockwise."
            )
    if len(np.unique(np.sort(az)[1:] - np.sort(az)[:-1])) > 1:
        warn("The azimuth angles of the current dataset are not equidistant.")
    return r, az


def _is_sorted(x):
    """
    Returns True when array x is sorted
    """
    return np.all(x[:-1] <= x[1:])


def _get_range_resolution(x):
    """
    Returns the range resolution based on
    the array x of the range gates' exterior limits
    """
    if len(x) <= 1:
        raise ValueError(
            "The range gate array has to contain at least "
            "two values for deriving the resolution."
        )
    res = np.unique(x[1:] - x[:-1])
    if len(res) > 1:
        raise ValueError("The resolution of the range array is ambiguous.")
    return res[0]


def _get_azimuth_resolution(x):
    """
    Returns the azimuth resolution based on the array x of the beams'
    azimuth angles
    """
    res = np.unique(np.sort(x)[1:] - np.sort(x)[:-1])
    if len(res) > 1:
        raise ValueError("The resolution of the azimuth angle array is ambiguous.")
    return res[0]


def sweep_centroids(nrays, rscale, nbins, elangle, /):
    """Construct sweep centroids native coordinates.

    Parameters
    ----------
    nrays : int
        number of rays
    rscale : float
        length [m] of a range bin
    nbins : int
        number of range bins
    elangle : float
        elevation angle [deg]

    Returns
    -------
    coordinates : :py:class:`numpy:numpy.ndarray`
        array of shape (nrays,nbins,3) containing native centroid radar
        coordinates (slant range, azimuth, elevation)
    """
    ascale = 360.0 / nrays
    azimuths = ascale / 2.0 + np.linspace(0, 360.0, nrays, endpoint=False)
    ranges = np.arange(nbins) * rscale + rscale / 2.0
    coordinates = np.empty((nrays, nbins, 3), dtype=float)
    coordinates[:, :, 0] = np.tile(ranges, (nrays, 1))
    coordinates[:, :, 1] = np.transpose(np.tile(azimuths, (nbins, 1)))
    coordinates[:, :, 2] = elangle
    return coordinates


def maximum_intensity_projection(
    data, *, r=None, az=None, angle=None, elev=None, autoext=True
):
    """Computes the maximum intensity projection along an arbitrary cut \
    through the ppi from polar data.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array containing polar data (azimuth, range)
    r : :class:`numpy:numpy.ndarray`, optional
        Array containing range data
    az : :class:`numpy:numpy.ndarray`, optional
        Array containing azimuth data
    angle : float, optional
        angle of slice, Defaults to 0. Should be between 0 and 180.
        0. means horizontal slice, 90. means vertical slice
    elev : float, optional
        elevation angle of scan, Defaults to 0.
    autoext : bool, optional
        This routine uses :func:`numpy.numpy.digitize` to bin the data.
        As this function needs bounds, we create one set of coordinates more
        than would usually be provided by `r` and `az`. Defaults to True.

    Returns
    -------
    xs : :class:`numpy:numpy.ndarray`
        meshgrid x array
    ys : :class:`numpy:numpy.ndarray`
        meshgrid y array
    mip : :class:`numpy:numpy.ndarray`
        Array containing the maximum intensity projection (range, range*2)
    """
    # providing 'reasonable defaults', based on the data's shape
    if r is None:
        r = np.arange(data.shape[1], dtype=np.float_)
    if az is None:
        az = np.arange(data.shape[0], dtype=np.float_)

    if angle is None:
        angle = 0.0

    if elev is None:
        elev = 0.0

    if autoext:
        # the ranges need to go 'one bin further', assuming some regularity
        # we extend by the distance between the preceding bins.
        x = np.append(r, r[-1] + (r[-1] - r[-2]))
        # the angular dimension is supposed to be cyclic, so we just add the
        # first element
        y = np.append(az, az[0])
    else:
        # no autoext basically is only useful, if the user supplied the correct
        # dimensions himself.
        x = r
        y = az

    # roll data array to specified azimuth, assuming equidistant azimuth angles
    ind = (az >= angle).nonzero()[0][0]
    data = np.roll(data, ind, axis=0)

    # build cartesian range array, add delta to last element to compensate for
    # open bound (np.digitize)
    dc = np.linspace(-np.max(r), np.max(r) + 0.0001, num=r.shape[0] * 2 + 1)

    # get height values from polar data and build cartesian height array
    # add delta to last element to compensate for open bound (np.digitize)
    hp = np.zeros((y.shape[0], x.shape[0]))
    hc = misc.bin_altitude(x, elev, 0, re=6370040.0)
    hp[:] = hc
    hc[-1] += 0.0001

    # create meshgrid for polar data
    xx, yy = np.meshgrid(x, y)

    # create meshgrid for cartesian slices
    xs, ys = np.meshgrid(dc, hc)

    # convert polar coordinates to cartesian
    xxx = xx * np.cos(np.radians(90.0 - yy))

    # digitize coordinates according to cartesian range array
    range_dig1 = np.digitize(xxx.ravel(), dc)
    range_dig1.shape = xxx.shape

    # digitize heights according polar height array
    height_dig1 = np.digitize(hp.ravel(), hc)
    # reshape accordingly
    height_dig1.shape = hp.shape

    # what am I doing here?!
    range_dig1 = range_dig1[0:-1, 0:-1]
    height_dig1 = height_dig1[0:-1, 0:-1]

    # create height and range masks
    height_mask = [(height_dig1 == i).ravel().nonzero()[0] for i in range(1, len(hc))]
    range_mask = [(range_dig1 == i).ravel().nonzero()[0] for i in range(1, len(dc))]

    # create mip output array, set outval to inf
    mip = np.zeros((r.shape[0], 2 * r.shape[0]))
    mip[:] = np.inf

    # fill mip array,
    # in some cases there are no values found in the specified range and height
    # then we fill in nans and interpolate
    for i in range(0, len(range_mask)):
        mask1 = range_mask[i]
        found = False
        for j in range(0, len(height_mask)):
            mask2 = np.intersect1d(mask1, height_mask[j])
            # this is to catch the ValueError from the max() routine when
            # calculating on empty array
            try:
                mip[j, i] = data.ravel()[mask2].max()
                if not found:
                    found = True
            except ValueError:
                if found:
                    mip[j, i] = np.nan

    # interpolate nans inside image, do not touch outvals
    good = ~np.isnan(mip)
    xp = good.ravel().nonzero()[0]
    fp = mip[~np.isnan(mip)]
    x = np.isnan(mip).ravel().nonzero()[0]
    mip[np.isnan(mip)] = np.interp(x, xp, fp)

    # reset outval to nan
    mip[mip == np.inf] = np.nan

    return xs, ys, mip


def georeference(obj, **kwargs):
    """Georeference Dataset/DataArray.

        .. versionadded:: 1.5

    This function adds georeference data to xarray Dataset/DataArray `obj`.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.Dataset` or :py:class:`xarray:xarray.DataArray`

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`, :py:class:`cartopy.crs.CRS` or None
        If GDAL OSR SRS, output is in this projection, else AEQD.
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    ----------
    obj : :py:class:`xarray:xarray.Dataset` or :py:class:`xarray:xarray.DataArray`
    """
    trg_crs = kwargs.pop("crs", "None")
    re = kwargs.pop("re", None)
    ke = kwargs.pop("ke", 4.0 / 3.0)

    # adding xyz aeqd-coordinates
    site = (
        obj.coords["longitude"].values,
        obj.coords["latitude"].values,
        obj.coords["altitude"].values,
    )

    if site == (0.0, 0.0, 0.0):
        re = 6378137.0

    # create meshgrid to overcome dimension problem with spherical_to_xyz
    r, az = np.meshgrid(obj["range"], obj["azimuth"])

    # GDAL OSR, convert to this crs
    if has_import(osr) and isinstance(trg_crs, osr.SpatialReference):
        xyz = spherical_to_proj(
            r, az, obj["elevation"], site, crs=trg_crs, re=re, ke=ke
        )
    # other crs, convert to aeqd
    elif trg_crs:
        xyz, trg_crs = spherical_to_xyz(
            r, az, obj["elevation"], site, re=re, ke=ke, squeeze=True
        )
    # crs, convert to aeqd and add offset
    else:
        xyz, trg_crs = spherical_to_xyz(
            r, az, obj["elevation"], site, re=re, ke=ke, squeeze=True
        )
        xyz += np.array(site).T

    # calculate center point
    # use first range bins
    ax = tuple(range(xyz.ndim - 2))
    center = np.mean(xyz[..., 0, :], axis=ax)

    # calculate ground range
    gr = np.sqrt((xyz[..., 0] - center[0]) ** 2 + (xyz[..., 1] - center[1]) ** 2)

    # dimension handling
    dim0 = obj["azimuth"].dims[-1]
    if obj["elevation"].dims:
        dimlist = list(obj["elevation"].dims)
    else:
        dimlist = list(obj["azimuth"].dims)

    # xyz is an array of cartesian coordinates for every spherical coordinate,
    # so the possible dimensions are: elevation, azimuth, range, 3.
    # For 2d, it either has (elevation, range, 3) or (azimuth, range, 3) dimensions.
    # For 3d, the only option is the full (elevation, azimuth, range, 3) dimensions.
    # Thus, adding the following two lines for the 3d case should not break other functionalities,
    # and there should not be a case with more than 3 dimensions
    if xyz.ndim > 3:
        dimlist += ["azimuth"]

    dimlist += ["range"]

    # add xyz, ground range coordinates
    x_attrs = {"standard_name": "east_west_distance_from_radar", "units": "meters"}
    y_attrs = {"standard_name": "north_south_distance_from_radar", "units": "meters"}
    z_attrs = {"standard_name": "height_above_ground", "units": "meters"}
    gr_attrs = {"standard_name": "distance_from_radar", "units": "meters"}
    obj.coords["x"] = (dimlist, xyz[..., 0], x_attrs)
    obj.coords["y"] = (dimlist, xyz[..., 1], y_attrs)
    obj.coords["z"] = (dimlist, xyz[..., 2], z_attrs)
    obj.coords["gr"] = (dimlist, gr, gr_attrs)

    # adding rays, bins coordinates
    if obj.sweep_mode == "azimuth_surveillance":
        bins, rays = np.meshgrid(obj["range"], obj["azimuth"], indexing="xy")
    else:
        bins, rays = np.meshgrid(obj["range"], obj["elevation"], indexing="xy")
    obj.coords["rays"] = ([dim0, "range"], rays, obj[dim0].attrs)
    obj.coords["bins"] = ([dim0, "range"], bins, obj["range"].attrs)

    # convert GDAL OSR to WKT
    if has_import(osr):
        trg_crs = trg_crs.ExportToWkt(["FORMAT=WKT2_2018"])

    # import into pyproj CRS
    proj_crs = pyproj.CRS.from_user_input(trg_crs)
    obj = add_crs(obj, crs=proj_crs)

    return obj


class GeorefPolarMethods:
    """wradlib xarray SubAccessor methods for Georef Polar Methods."""

    @docstring(georeference)
    def georeference(self, *args, **kwargs):
        if not isinstance(self, GeorefPolarMethods):
            return georeference(self, *args, **kwargs)
        else:
            return georeference(self._obj, *args, **kwargs)

    @docstring(_spherical_to_xyz_xarray)
    def spherical_to_xyz(self, *args, **kwargs):
        if not isinstance(self, GeorefPolarMethods):
            return spherical_to_xyz(self, *args, **kwargs)
        else:
            return spherical_to_xyz(self._obj, *args, **kwargs)

    @docstring(_spherical_to_proj_xarray)
    def spherical_to_proj(self, *args, **kwargs):
        if not isinstance(self, GeorefPolarMethods):
            return spherical_to_proj(self, *args, **kwargs)
        else:
            return spherical_to_proj(self._obj, *args, **kwargs)

    @docstring(_spherical_to_polyvert_xarray)
    def spherical_to_polyvert(self, *args, **kwargs):
        if not isinstance(self, GeorefPolarMethods):
            return spherical_to_polyvert(self, *args, **kwargs)
        else:
            return spherical_to_polyvert(self._obj, *args, **kwargs)

    @docstring(_spherical_to_centroids_xarray)
    def spherical_to_centroids(self, *args, **kwargs):
        if not isinstance(self, GeorefPolarMethods):
            return spherical_to_centroids(self, *args, **kwargs)
        else:
            return spherical_to_centroids(self._obj, *args, **kwargs)
