#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Polar Grid Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   spherical_to_xyz
   spherical_to_proj
   spherical_to_polyvert
   spherical_to_centroids
   centroid_to_polyvert
   sweep_centroids

"""

import numpy as np
import warnings

import wradlib.georef as georef
import wradlib.ipol as ipol
import wradlib.util as util


def spherical_to_xyz(r, phi, theta, sitecoords, re=None, ke=4./3.,
                     squeeze=None, strict_dims=False):
    """Transforms spherical coordinates (r, phi, theta) to cartesian
    coordinates (x, y, z) centered at sitecoords (aeqd).

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
    sitecoords : a sequence of three floats
        the lon / lat coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths.
    squeeze : bool
        If True, returns squeezed array.
    strict_dims : bool
        If True, generates output of (theta, phi, r, 3) in any case.
        If False, dimensions with same length are "merged".

    Returns
    -------
    xyz : :class:`numpy:numpy.ndarray`
        Array of shape (..., 3). Contains cartesian coordinates.
    rad : osr object
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    """
    # if site altitude is present, use it, else assume it to be zero
    try:
        centalt = sitecoords[2]
    except IndexError:
        centalt = 0.

    # if no radius is given, get the approximate radius of the WGS84
    # ellipsoid for the site's latitude
    proj4_to_osr = georef.proj4_to_osr
    if re is None:
        re = georef.get_earth_radius(sitecoords[1])
        # Set up aeqd-projection sitecoord-centered, wgs84 datum and ellipsoid
        # use world azimuthal equidistant projection
        rad = proj4_to_osr(('+proj=aeqd +lon_0={lon:f} +x_0=0 +y_0=0 ' +
                            '+lat_0={lat:f} +ellps=WGS84 +datum=WGS84 ' +
                            '+units=m +no_defs').format(lon=sitecoords[0],
                                                        lat=sitecoords[1]))
    else:
        # Set up aeqd-projection sitecoord-centered, assuming spherical earth
        # use Sphere azimuthal equidistant projection
        rad = proj4_to_osr(('+proj=aeqd +lon_0={lon:f} ' +
                            '+lat_0={lat:f} +a={a:f} +b={b:f} ' +
                            '+units=m +no_defs').format(lon=sitecoords[0],
                                                        lat=sitecoords[1],
                                                        a=re, b=re))

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

    z = georef.bin_altitude(r, theta, centalt, re, ke=ke)
    dist = georef.site_distance(r, theta, z, re, ke=ke)

    if ((not strict_dims) and phi.ndim and r.ndim and
            (r.shape[2] == phi.shape[1])):
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

    if squeeze is None:
        warnings.warn("Function `spherical_to_xyz` returns an array "
                      "of shape (theta, phi, range, 3). Use `squeeze=True` "
                      "to remove singleton dimensions."
                      "", DeprecationWarning)
    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz, rad


def spherical_to_proj(r, phi, theta, sitecoords, proj=None, re=None, ke=4./3.):
    """Transforms spherical coordinates (r, phi, theta) to projected
    coordinates centered at sitecoords in given projection.

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
    sitecoords : a sequence of three floats
        the lon / lat coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    proj : osr object
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
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
    >>> csite = (9.0, 48.0)
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
    if proj is None:
        proj = georef.get_default_projection()

    xyz, rad = spherical_to_xyz(r, phi, theta, sitecoords, re=re, ke=ke,
                                squeeze=True)

    # reproject aeqd to destination projection
    coords = georef.reproject(xyz, projection_source=rad,
                              projection_target=proj)

    return coords


def centroid_to_polyvert(centroid, delta):
    """Calculates the 2-D Polygon vertices necessary to form a rectangular
    polygon around the centroid's coordinates.

    The vertices order will be clockwise, as this is the convention used
    by ESRI's shapefile format for a polygon.

    Parameters
    ----------
    centroid : array_like
               List of 2-D coordinates of the center point of the rectangle.
    delta :    scalar or :class:`numpy:numpy.ndarray`
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
        raise ValueError("Parameter 'centroid' dimensions need "
                         "to be (..., 2)")
    dshape = [1] * cent.ndim
    dshape.insert(-1, 5)
    dshape[-1] = 2

    d = np.array([[-1., -1.],
                  [-1., 1.],
                  [1., 1.],
                  [1., -1.],
                  [-1., -1.]]).reshape(tuple(dshape))

    return np.asanyarray(centroid)[..., None, :] + d * np.asanyarray(delta)


def spherical_to_polyvert(r, phi, theta, sitecoords, proj=None):
    """
    Generate 3-D polygon vertices directly from spherical coordinates
    (r, phi, theta).

    This is an alternative to :func:`~wradlib.georef.centroid_to_polyvert`
    which does not use centroids, but generates the polygon vertices by simply
    connecting the corners of the radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    :func:`~wradlib.georef.spherical_to_xyz`.

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
    sitecoords : a sequence of three floats
        the lon/lat/alt coordinates of the radar location
    proj : osr object
        Destination Projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape(num_vertices,
        num_vertex_nodes, 2). The last dimension carries the xyz-coordinates
        either in `aeqd` or given proj.
    proj : osr object
        only returned if proj is None

    Examples
    --------
    >>> import wradlib.georef as georef  # noqa
    >>> import numpy as np
    >>> from matplotlib import collections
    >>> import matplotlib.pyplot as pl
    >>> #pl.interactive(True)
    >>> # define the polar coordinates and the site coordinates in lat/lon
    >>> r = np.array([50., 100., 150., 200.]) * 1000
    >>> # _check_polar_coords fails in next line
    >>> # az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])
    >>> az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
    >>> el = 1.0
    >>> sitecoords = (9.0, 48.0, 0)
    >>> polygons, proj = georef.spherical_to_polyvert(r, az, el, sitecoords)
    >>> # plot the resulting mesh
    >>> fig = pl.figure()
    >>> ax = fig.add_subplot(111)
    >>> #polycoll = mpl.collections.PolyCollection(vertices,closed=True, facecolors=None)  # noqa
    >>> polycoll = collections.PolyCollection(polygons[...,:2], closed=True, facecolors='None')  # noqa
    >>> ret = ax.add_collection(polycoll, autolim=True)
    >>> pl.autoscale()
    >>> pl.show()

    """
    # prepare the range and azimuth array so they describe the boundaries of
    # a bin, not the centroid
    r, phi = _check_polar_coords(r, phi)
    r = np.insert(r, 0, r[0] - _get_range_resolution(r))
    phi = phi - 0.5 * _get_azimuth_resolution(phi)
    phi = np.append(phi, phi[0])
    phi = np.where(phi < 0, phi + 360., phi)

    # generate a grid of polar coordinates of bin corners
    r, phi = np.meshgrid(r, phi)

    coords, rad = spherical_to_xyz(r, phi, theta, sitecoords, squeeze=True,
                                   strict_dims=True)
    if proj is not None:
        coords = georef.reproject(coords, projection_source=rad,
                                  projection_target=proj)

    llc = coords[:-1, :-1]
    ulc = coords[:-1, 1:]
    urc = coords[1:, 1:]
    lrc = coords[1:, :-1]

    vertices = np.stack((llc, ulc, urc, lrc, llc), axis=-2).reshape((-1, 5, 3))

    if proj is None:
        return vertices, rad
    else:
        return vertices


def spherical_to_centroids(r, phi, theta, sitecoords, proj=None):
    """
    Generate 3-D centroids of the radar bins from the sperical
    coordinates (r, phi, theta).

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. The ranges are assumed to define the exterior
    boundaries of the range bins (thus they must be positive). The angles are
    assumed to describe the pointing direction fo the main beam lobe.

    For further information refer to the documentation of
    :meth:`~wradlib.georef.polar2lonlat`.

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
    sitecoords : a sequence of three floats
        the lon/lat/alt coordinates of the radar location
    proj : osr object
        Destination Projection

    Returns
    -------
    output : centroids :class:`numpy:numpy.ndarray`
        A 3-d array of bin centroids with shape(num_rays, num_bins, 3).
        The last dimension carries the xyz-coordinates
        either in `aeqd` or given proj.
    proj : osr object
        only returned if proj is None

    Note
    ----
    Azimuth angles of 360 deg are internally converted to 0 deg.

    """
    # make sure the range and azimuth angles have the right properties
    r, phi = _check_polar_coords(r, phi)

    r = r - 0.5 * _get_range_resolution(r)

    # generate a polar grid and convert to lat/lon
    r, phi = np.meshgrid(r, phi)

    coords, rad = spherical_to_xyz(r, phi, theta, sitecoords, squeeze=True)

    if proj is None:
        return coords, rad
    else:
        return georef.reproject(coords, projection_source=rad,
                                projection_target=proj)


def _check_polar_coords(r, az):
    """
    Contains a lot of checks to make sure the polar coordinates are adequate.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        range gates in any unit
    az : :class:`numpy:numpy.ndarray`
        azimuth angles in degree

    """
    r = np.array(r, 'f4')
    az = np.array(az, 'f4')
    az[az == 360.] = 0.
    if 0. in r:
        raise ValueError('Invalid polar coordinates: '
                         '0 is not a valid range gate specification '
                         '(the centroid of a range gate must be positive).')
    if len(np.unique(r)) != len(r):
        raise ValueError('Invalid polar coordinates: '
                         'Range gate specification contains duplicate '
                         'entries.')
    if len(np.unique(az)) != len(az):
        raise ValueError('Invalid polar coordinates: '
                         'Azimuth specification contains duplicate entries.')
    if not _is_sorted(r):
        raise ValueError('Invalid polar coordinates: '
                         'Range array must be sorted.')
    if len(np.unique(r[1:] - r[:-1])) > 1:
        raise ValueError('Invalid polar coordinates: '
                         'Range gates are not equidistant.')
    if len(np.where(az >= 360.)[0]) > 0:
        raise ValueError('Invalid polar coordinates: '
                         'Azimuth angles must not be greater than '
                         'or equal to 360 deg.')
    if not _is_sorted(az):
        # it is ok if the azimuth angle array is not sorted, but it has to be
        # 'continuously clockwise', e.g. it could start at 90° and stop at °89
        az_right = az[np.where(np.logical_and(az <= 360, az >= az[0]))[0]]
        az_left = az[np.where(az < az[0])]
        if (not _is_sorted(az_right)) or (not _is_sorted(az_left)):
            raise ValueError('Invalid polar coordinates: '
                             'Azimuth array is not sorted clockwise.')
    if len(np.unique(np.sort(az)[1:] - np.sort(az)[:-1])) > 1:
        warnings.warn("The azimuth angles of the current "
                      "dataset are not equidistant.", UserWarning)
        # print('Invalid polar coordinates: Azimuth angles '
        #       'are not equidistant.')
        # exit()
    return r, az


def _is_sorted(x):
    """
    Returns True when array x is sorted
    """
    return np.all(x == np.sort(x))


def _get_range_resolution(x):
    """
    Returns the range resolution based on
    the array x of the range gates' exterior limits
    """
    if len(x) <= 1:
        raise ValueError('The range gate array has to contain at least '
                         'two values for deriving the resolution.')
    res = np.unique(x[1:] - x[:-1])
    if len(res) > 1:
        raise ValueError('The resolution of the range array is ambiguous.')
    return res[0]


def _get_azimuth_resolution(x):
    """
    Returns the azimuth resolution based on the array x of the beams'
    azimuth angles
    """
    res = np.unique(np.sort(x)[1:] - np.sort(x)[:-1])
    if len(res) > 1:
        raise ValueError('The resolution of the azimuth angle array '
                         'is ambiguous.')
    return res[0]


def sweep_centroids(nrays, rscale, nbins, elangle):
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
        elevation angle [radians]

    Returns
    -------
    coordinates : 3d array
        array of shape (nrays,nbins,3) containing native centroid radar
        coordinates (slant range, azimuth, elevation)
    """
    ascale = 2 * np.pi / nrays
    azimuths = ascale / 2. + np.linspace(0, 2 * np.pi, nrays, endpoint=False)
    ranges = np.arange(nbins) * rscale + rscale / 2.
    coordinates = np.empty((nrays, nbins, 3), dtype=float)
    coordinates[:, :, 0] = np.tile(ranges, (nrays, 1))
    coordinates[:, :, 1] = np.transpose(np.tile(azimuths, (nbins, 1)))
    coordinates[:, :, 2] = elangle
    return coordinates


def polar_to_cart(coords):
    """Converts polar coordinates to cartesian coordinates
       Uses the radar convention (starting north and going clockwise)

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains polar coordinates (r,a).

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains cartesian coordinates (x, y).
    """
    ranges = coords[..., 0]
    azimuths = coords[..., 1]

    x = ranges * np.cos(np.radians(90 - azimuths))
    y = ranges * np.sin(np.radians(90 - azimuths))

    coords = np.stack((x, y), axis=-1)

    return(coords)


def cart_to_polar(coords):
    """Convert cartesian coordinates to polar coordinates
       Uses the radar convention (starting north and going clockwise)

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains cartesian coordinates (x, y).

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2). Contains polar coordinates (r,a).
    """
    x = coords[..., 0]
    y = coords[..., 1]

    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    t = t*180/np.pi
    t = 90 - t
    t[t < 0] = 360 + t[t < 0]

    coords = np.stack((r, t), axis=-1)

    return(coords)


def bin_gcz(height_radar, elangle, ranges, re=None, ke=4/3):
    """Calculate great circle distance and height of a radar bin.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    if re is None:
        re = 6371007.0

    gamma = np.deg2rad(90 + elangle)

    rk = re*ke

    a = ranges
    b = rk + height_radar
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(gamma))
    height = c - rk

    alpha = np.arcsin(a * np.sin(gamma) / c)
    distance = rk * alpha

    return distance, height


def bin_gcz_doviak(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point (Doviak, Zrnic).

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    gamma = np.deg2rad(elangle)

    # four third radius model for refraction
    rk = (re * ke)

    a = ranges
    b = rk
    c = np.sqrt(a**2 + b**2 + 2 * a * b * np.sin(gamma))
    height = c - rk

    distance = rk * np.arcsin(a * np.cos(gamma) / (b + height))

    height = height + height_radar

    return distance, height


def bin_gcz_other(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth


    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    gamma = np.deg2rad(elangle)

    rk = re * ke

    a = ranges
    b = rk + height_radar
    c = np.sqrt(a**2 + b**2 + 2 * a * b * np.sin(gamma))
    height = c - rk

    tmp1 = a * np.cos(gamma)
    tmp2 = a * np.sin(gamma) + b
    distance = rk * np.arctan(tmp1 / tmp2)

    return distance, height


def bin_gcz_full(height_radar, elangle, ranges, re=6371007.0, ke=4/3):
    """Calculate on ground distance and height of target point.

    Parameters
    ----------
    height_radar : float
        height of the radar above sphere
    elangle : float
        elevation angle
    ranges : array of float
        distance along the radar beam
    re : float
        radius of the Earth

    Returns
    -------
    distance : float
        great circle distance from the radar
    height : float
        height above sphere

    """

    elangle = np.deg2rad(elangle)

    # four third radius model for refraction
    rk = (re * ke)

    # radius of radar beam curvature
    rc = 1 / (np.cos(elangle) * (ke-1) / (rk))

    # euclidian distance from radar to bin
    alpha = ranges / rc
    distance = 2 * rc * np.sin(alpha/2)

    # height from sphere (cosine rule)
    gamma = np.pi/2 - alpha/2 + elangle
    a = height_radar + re
    b = distance
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    rbin = c
    height = rbin - re

    # arc distance over sphere (sin rule)
    beta = np.arcsin(b * np.sin(gamma) / c)
    arc_distance = re * beta

    return arc_distance, height


def sweep_coordinates(ranges, azimuths):
    """Get the sweep coordinates

    Parameters
    ----------
    ranges : :class:`numpy:numpy.array`
        Contains the radial distances in meters.
    azimuths : :class:`numpy:numpy.array`
        Contains the azimuthal angles in degree.

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrays, nbins, 2) containing the sweep coordinates.

    """

    ranges, azimuths = np.meshgrid(ranges, azimuths)
    sweepcoords = np.stack((ranges, azimuths), axis=-1)

    return(sweepcoords)


def sweep_to_map(sitecoords, elangle, ranges, azimuths,
                 projection=None, altitude=False,
                 binsize=None):
    """Get map coordinates from sweep coordinates

    Parameters
    ----------
    sitecoords : array of floats
        radar site location: longitude, latitude and altitude (amsl)
    elangle: :class:`numpy:numpy.array`
        Contains the elevation angle in degree.
    ranges : :class:`numpy:numpy.array`
        Contains the radial distances in meters.
    azimuths : :class:`numpy:numpy.array`
        Contains the azimuthal angles in degree.
    projection : osr.SpatialReference
        map projection definition
    altitude : bool
        True to get also the altitude
        for the given projection

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrays, nbins, ndim) containing the map coordinates.
        If binsize is not None, an array

    """
    sitecoords = georef.geoid_to_ellipsoid(sitecoords)

    re = georef.get_earth_radius(sitecoords[1])

    ranges, z = georef.bin_gcz(sitecoords[2], elangle, ranges, re)

    ranges, azimuths = np.meshgrid(ranges, azimuths)
    coords = np.stack((ranges, azimuths), axis=-1)
    coords = georef.polar_to_cart(coords)

    if altitude:
        x = coords[..., 0]
        y = coords[..., 1]
        z = np.tile(z, (x.shape[0], 1))
        coords = np.stack((x, y, z), axis=-1)

    # Reproject if needed
    if projection is not None:
        radar = georef.get_radar_projection(sitecoords)
        coords = georef.reproject(coords, projection_source=radar,
                                  projection_target=projection)

    return coords


def sweep_to_polyvert(*args, binsize=None, ravel=False, **kwargs):
    """Get projected polygon representation of sweep bins

    Parameters
    ----------
    args :
        arguments for sweep_to_map
    binsize : tuple of float
        bin size (range, azimuth)
    kwargs :
        keyword arguments for sweep_to_map
    ravel :
        True to ravel

    Returns
    -------
    vertices : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape(nbins, 5, 2).

    """
    ranges = args[2]
    azimuths = args[3]
    if binsize is None:
        rscale = (ranges[1] - ranges[0])/2
        ascale = (azimuths[1] - azimuths[0])/2
    else:
        rscale, ascale = binsize

    rsta = ranges - rscale
    rend = ranges + rscale
    asta = azimuths - ascale
    aend = azimuths + ascale

    args = args[0:2]

    ulc = sweep_to_map(*args, rsta, asta, **kwargs)
    urc = sweep_to_map(*args, rend, asta, **kwargs)
    lrc = sweep_to_map(*args, rend, aend, **kwargs)
    llc = sweep_to_map(*args, rsta, aend, **kwargs)

    vertices = np.stack((ulc, urc, lrc, llc, ulc), axis=-2)

    if ravel:
        vertices = vertices.reshape((-1, 5, 2))

    return(vertices)


def map_to_sweep(coords, sitecoords, elangle, projection=None):
    """Returns sweep coordinates from map coordinates.

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        array of shape (..., ndim) containing map coordinates.
    sitecoords : array of floats
        radar site location: longitude, latitude and altitude (amsl)
    elangle: :class:`numpy:numpy.array`
        elevation angle in degree.
    projection : osr object
        map projection definition

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., 2) containing the sweep coordinates.
    """

    coords = coords[..., 0:2]
    sitecoords = georef.geoid_to_ellipsoid(sitecoords)
    if projection is not None:
        radar = georef.get_radar_projection(sitecoords)
        coords = georef.reproject(coords,
                                  projection_source=projection,
                                  projection_target=radar)
    # cartesian to polar
    coords = georef.cart_to_polar(coords)
    dists = coords[..., 0]
    azimuths = coords[..., 1]

    re = 6371007.0
    ke = 4/3
    rk = re * ke

    alpha = dists/rk
    gamma = np.pi/2 + np.deg2rad(elangle)
    beta = np.pi - alpha - gamma
    b = rk + sitecoords[2]
    ranges = b * np.sin(alpha)/np.sin(beta)

    coords = np.stack((ranges, azimuths), axis=-1)

    return coords


def raster_to_sweep(rastercoords, projection,
                    sitecoords, elangle, ranges, azimuths,
                    binsize=None, rastervalues=None,
                    method='area', fill='nearest',
                    **kwargs):
    """
    Map raster values to a radar sweep
    taking into account scale differences.

    Parameters
    ----------
    rastercoords : numpy ndarray
        raster coordinates
    projection : osr.SpatialReference
        raster projection definition
    sitecoords : sequence of 3 floats
        Longitude, Latitude and Altitude (in meters above sea level)
    elangle : float
        elevation angle
    ranges : numpy array
        range coordinates
    azimuths : numpy array
        azimuth coordinates
    binsize : (float, float)
        bin size in range and azimuth
    rastervalues : numpy ndarray
        raster values
    method : string
        interpolation method: nearest, linear, spline, binned or area
    fill : string
        second method to fill holes: nearest, linear, spline
    kwargs : keyword arguments
        keyword arguments to interpolation class

    Returns
    -------
    sweepval : numpy ndarray
        sweep values of size (nrays, nbins)
    """

    if method not in ["nearest", "linear", "spline", "binned", "area"]:
        raise ValueError("Invalid method")

    if binsize is not None and method != "area":
        raise ValueError("Only 'area' method supported with bin size")

    if method in ["linear", "nearest", "spline"]:

        fill = None

        sweepcoords = sweep_to_map(sitecoords, elangle, ranges, azimuths,
                                   projection)

        if method == "spline":
            interpolator = ipol.RectSpline(rastercoords, sweepcoords,
                                           **kwargs)
        else:
            interpolator = ipol.RectLinear(rastercoords, sweepcoords,
                                           method, **kwargs)

    if method == "binned":
        rastercoords_sweep = map_to_sweep(rastercoords, sitecoords, elangle,
                                          projection)
        sweepcoords = sweep_coordinates(ranges, azimuths)
        interpolator = ipol.RectBin(rastercoords_sweep, sweepcoords)

    if method == "area":

        edges = util.grid_center_to_edge(rastercoords)
        pixels = util.grid_to_polyvert(edges)

        bins = georef.sweep_to_polyvert(sitecoords, elangle,
                                        ranges, azimuths, binsize,
                                        projection=projection)

        interpolator = ipol.PolyArea(pixels, bins)

    if fill is not None:
        alternative = raster_to_sweep(rastercoords, projection,
                                      sitecoords, elangle, ranges, azimuths,
                                      binsize,
                                      method=fill,
                                      **kwargs)
        interpolator = ipol.Sequence([interpolator, alternative])

    if rastervalues is None:
        return(interpolator)

    sweepval = interpolator(rastervalues)

    return(sweepval)


def raster_to_sweep_multi(rasters, *args, **kwargs):
    """
    Map several rasters to a radar sweep

    Parameters
    ----------
    raster : list of gdal.Dataset
        georeferenced raster images
    args : arguments
        passed to raster_to_sweep
    kwargs : keyword arguments
        passed to raster_to_sweep
    """
    sweepval = None

    for raster in rasters:
        rastervalues = georef.read_gdal_values(raster)
        rastercoords = georef.read_gdal_coordinates(raster)
        projection = georef.read_gdal_projection(raster)
        temp = raster_to_sweep(rastercoords, projection,
                               *args, rastervalues=rastervalues, **kwargs)
        if sweepval is None:
            sweepval = temp
            continue
        bad = np.isnan(temp)
        sweepval[~bad] = temp[~bad]

    return(sweepval)


def sweep_to_raster(rastercoords, projection,
                    sitecoords, elangle, ranges, azimuths,
                    binsize=None, sweepvalues=None,
                    method='area', fill='nearest',
                    **kwargs):
    """
    Map sweep bin values to raster cells

    Parameters
    ----------
    rastercoords : numpy ndarray
        raster coordinates
    projection : osr.SpatialReference
        raster projection definition
    sitecoords : array of floats
        Longitude, Latitude and Altitude (in meters above sea level)
    elangle : float
        elevation angle
    ranges : numpy array
        sweep ranges bin center
    azimuths : numpy array
        sweep azimuth edges
    binsize : float
        bin size in range and azimuth
    sweepvalues : numpy ndarray
        sweep values
    method : string
        interpolation method: nearest, linear, spline, binned or area
    fill : string
        second method to fill holes: nearest, linear, spline
    kwargs : keyword arguments
        keyword arguments to raster_to_sweep_method

    Returns
    -------
    rastervalues : numpy ndarray
        rastervalues values of size (nrows, ncols)
    """

    if method not in ["nearest", "linear", "spline", "binned", "area"]:
        raise ValueError("Invalid method")

    if binsize is not None and method != "area":
        raise ValueError("Only 'area' method supported with bin size")

    if method in ["linear", "nearest", "spline"]:

        fill = None

        myrastercoords = map_to_sweep(rastercoords, sitecoords, elangle,
                                      projection)
        sweepcoords = sweep_coordinates(ranges, azimuths)

        if method == "spline":
            interpolator = ipol.RectSpline(sweepcoords, myrastercoords,
                                           **kwargs)
        else:
            interpolator = ipol.RectLinear(sweepcoords, myrastercoords,
                                           method, **kwargs)

    if method == "binned":

        sweepcoords = sweep_to_map(sitecoords, elangle, ranges, azimuths,
                                   projection)
        interpolator = ipol.RectBin(sweepcoords, rastercoords)

    if method == "area":

        edges = util.grid_center_to_edge(rastercoords)
        pixels = util.grid_to_polyvert(edges)

        bins = georef.sweep_to_polyvert(sitecoords, elangle,
                                        ranges, azimuths, binsize,
                                        projection=projection)

        interpolator = ipol.PolyArea(bins, pixels)

    if fill is not None:
        alternative = sweep_to_raster(rastercoords, projection,
                                      sitecoords, elangle, ranges, azimuths,
                                      binsize,
                                      method=fill,
                                      **kwargs)
        interpolator = ipol.Sequence([interpolator, alternative])

    if sweepvalues is None:
        return(interpolator)

    rastervalues = interpolator(sweepvalues)

    return(rastervalues)
