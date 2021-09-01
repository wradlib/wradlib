#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
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
    "spherical_to_xyz",
    "spherical_to_proj",
    "spherical_to_polyvert",
    "spherical_to_centroids",
    "centroid_to_polyvert",
    "sweep_centroids",
    "maximum_intensity_projection",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np

from wradlib.georef import misc, projection


def spherical_to_xyz(
    r, phi, theta, sitecoords, re=None, ke=4.0 / 3.0, squeeze=False, strict_dims=False
):
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
    sitecoords : sequence
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
        If True, returns squeezed array. Defaults to False.
    strict_dims : bool
        If True, generates output of (theta, phi, r, 3) in any case.
        If False, dimensions with same length are "merged".

    Returns
    -------
    xyz : :class:`numpy:numpy.ndarray`
        Array of shape (..., 3). Contains cartesian coordinates.
    rad : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Spatial Reference System (Projection).
        Defaults to wgs84 (epsg 4326).
    """
    # if site altitude is present, use it, else assume it to be zero
    try:
        centalt = sitecoords[2]
    except IndexError:
        centalt = 0.0

    # if no radius is given, get the approximate radius of the WGS84
    # ellipsoid for the site's latitude
    if re is None:
        re = projection.get_earth_radius(sitecoords[1])
        # Set up aeqd-projection sitecoord-centered, wgs84 datum and ellipsoid
        # use world azimuthal equidistant projection
        projstr = (
            "+proj=aeqd +lon_0={lon:f} +x_0=0 +y_0=0 +lat_0={lat:f} "
            + "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            + ""
        ).format(lon=sitecoords[0], lat=sitecoords[1])

    else:
        # Set up aeqd-projection sitecoord-centered, assuming spherical earth
        # use Sphere azimuthal equidistant projection
        projstr = (
            "+proj=aeqd +lon_0={lon:f} +lat_0={lat:f} +a={a:f} "
            "+b={b:f} +units=m +no_defs"
        ).format(lon=sitecoords[0], lat=sitecoords[1], a=re, b=re)

    rad = projection.proj4_to_osr(projstr)

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

    z = misc.bin_altitude(r, theta, centalt, re, ke=ke)
    dist = misc.site_distance(r, theta, z, re, ke=ke)

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

    return xyz, rad


def spherical_to_proj(r, phi, theta, sitecoords, proj=None, re=None, ke=4.0 / 3.0):
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
    sitecoords : sequence
        the lon / lat coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
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
        proj = projection.get_default_projection()

    xyz, rad = spherical_to_xyz(r, phi, theta, sitecoords, re=re, ke=ke, squeeze=True)

    # reproject aeqd to destination projection
    coords = projection.reproject(xyz, projection_source=rad, projection_target=proj)

    return coords


def centroid_to_polyvert(centroid, delta):
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
        raise ValueError("Parameter 'centroid' dimensions need " "to be (..., 2)")
    dshape = [1] * cent.ndim
    dshape.insert(-1, 5)
    dshape[-1] = 2

    d = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    ).reshape(tuple(dshape))

    return np.asanyarray(centroid)[..., None, :] + d * np.asanyarray(delta)


def spherical_to_polyvert(r, phi, theta, sitecoords, proj=None):
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
    sitecoords : sequence
        the lon/lat/alt coordinates of the radar location
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape(num_vertices,
        num_vertex_nodes, 2). The last dimension carries the xyz-coordinates
        either in `aeqd` or given proj.
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
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
    phi = np.where(phi < 0, phi + 360.0, phi)

    # generate a grid of polar coordinates of bin corners
    r, phi = np.meshgrid(r, phi)

    coords, rad = spherical_to_xyz(
        r, phi, theta, sitecoords, squeeze=True, strict_dims=True
    )
    if proj is not None:
        coords = projection.reproject(
            coords, projection_source=rad, projection_target=proj
        )

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
    sitecoords : sequence
        the lon/lat/alt coordinates of the radar location
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of bin centroids with shape(num_rays, num_bins, 3).
        The last dimension carries the xyz-coordinates
        either in `aeqd` or given proj.
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
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
        return projection.reproject(
            coords, projection_source=rad, projection_target=proj
        )


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
        raise ValueError("Invalid polar coordinates: " "Range array must be sorted.")
    if len(np.unique(r[1:] - r[:-1])) > 1:
        raise ValueError(
            "Invalid polar coordinates: " "Range gates are not equidistant."
        )
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
                "Invalid polar coordinates: " "Azimuth array is not sorted clockwise."
            )
    if len(np.unique(np.sort(az)[1:] - np.sort(az)[:-1])) > 1:
        warnings.warn(
            "The azimuth angles of the current " "dataset are not equidistant.",
            UserWarning,
        )
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
        raise ValueError("The resolution of the azimuth angle array " "is ambiguous.")
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
    data, r=None, az=None, angle=None, elev=None, autoext=True
):
    """Computes the maximum intensity projection along an arbitrary cut \
    through the ppi from polar data.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array containing polar data (azimuth, range)
    r : :class:`numpy:numpy.ndarray`
        Array containing range data
    az : :class:`numpy:numpy.ndarray`
        Array containing azimuth data
    angle : float
        angle of slice, Defaults to 0. Should be between 0 and 180.
        0. means horizontal slice, 90. means vertical slice
    elev : float
        elevation angle of scan, Defaults to 0.
    autoext : bool
        This routine uses numpy.digitize to bin the data.
        As this function needs bounds, we create one set of coordinates more
        than would usually be provided by `r` and `az`.

    Returns
    -------
    xs : :class:`numpy:numpy.ndarray`
        meshgrid x array
    ys : :class:`numpy:numpy.ndarray`
        meshgrid y array
    mip : :class:`numpy:numpy.ndarray`
        Array containing the maximum intensity projection (range, range*2)
    """
    # this may seem odd at first, but d1 and d2 are also used in several
    # plotting functions and thus it may be easier to compare the functions
    d1 = r
    d2 = az

    # providing 'reasonable defaults', based on the data's shape
    if d1 is None:
        d1 = np.arange(data.shape[1], dtype=np.float_)
    if d2 is None:
        d2 = np.arange(data.shape[0], dtype=np.float_)

    if angle is None:
        angle = 0.0

    if elev is None:
        elev = 0.0

    if autoext:
        # the ranges need to go 'one bin further', assuming some regularity
        # we extend by the distance between the preceding bins.
        x = np.append(d1, d1[-1] + (d1[-1] - d1[-2]))
        # the angular dimension is supposed to be cyclic, so we just add the
        # first element
        y = np.append(d2, d2[0])
    else:
        # no autoext basically is only useful, if the user supplied the correct
        # dimensions himself.
        x = d1
        y = d2

    # roll data array to specified azimuth, assuming equidistant azimuth angles
    ind = (d2 >= angle).nonzero()[0][0]
    data = np.roll(data, ind, axis=0)

    # build cartesian range array, add delta to last element to compensate for
    # open bound (np.digitize)
    dc = np.linspace(-np.max(d1), np.max(d1) + 0.0001, num=d1.shape[0] * 2 + 1)

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
    # xs, ys = np.meshgrid(dc,x)

    # convert polar coordinates to cartesian
    xxx = xx * np.cos(np.radians(90.0 - yy))
    # yyy = xx * np.sin(np.radians(90.-yy))

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
    mip = np.zeros((d1.shape[0], 2 * d1.shape[0]))
    mip[:] = np.inf

    # fill mip array,
    # in some cases there are no values found in the specified range and height
    # then we fill in nans and interpolate afterwards
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
