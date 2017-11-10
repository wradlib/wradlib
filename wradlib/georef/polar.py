#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Polar Grid Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   polar2lonlat
   polar2lonlatalt
   polar2lonlatalt_n
   polar2centroids
   polar2polyvert
   centroid2polyvert
   projected_bincoords_from_radarspecs
   sweep_centroids

"""

import numpy as np
import warnings

from .projection import proj4_to_osr, reproject
from .misc import (hor2aeq, beam_height_n, arc_distance_n,
                   get_earth_radius)


def _latscale(re=6370040.):
    """Return meters per degree latitude assuming spherical earth
    """
    return 2 * np.pi * re / 360.


def _lonscale(lat, re=6370040.):
    """Return meters per degree longitude assuming spherical earth
    """
    return (2 * np.pi * re / 360.) * np.cos(np.deg2rad(lat))


def polar2lonlat(r, az, sitecoords, re=6370040):
    """Transforms polar coordinates (of a PPI) to longitude/latitude \
    coordinates.

    This function assumes that the transformation from the polar radar
    coordinate system to the earth's spherical coordinate system may be done
    in the same way as astronomical observations are transformed from the
    horizon's coordinate system to the equatorial coordinate system.

    The conversion formulas used were taken from Wikipedia
    :cite:`Nautisches-Dreieck` and are only valid as long as the radar's
    elevation angle is small, as one main assumption of this method is, that
    the 'zenith-star'-side of the nautic triangle can be described by the radar
    range divided by the earths radius. For larger elevation angles, this side
    would have to be reduced.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    az : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location
    re : float
        earth's radius [m]

    Returns
    -------
    lon, lat : tuple
        Tuple of two :class:`numpy:numpy.ndarray` containing the spherical
        longitude and latitude coordinates.

    Note
    ----
    Be aware that the coordinates returned by this function are valid for
    a sphere. When using them in GIS make sure to distinguish that from
    the usually assumed WGS coordinate systems where the coordinates are based
    on a more complex ellipsoid.

    Examples
    --------

    A few standard directions (North, South, North, East, South, West) with
    different distances (amounting to roughly 1°) from a site
    located at 48°N 9°E

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])*1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> csite = (9.0, 48.0)
    >>> # csite = (0.0, 0.0)
    >>> lon1, lat1= polar2lonlat(r, az, csite)
    >>> for x, y in zip(lon1, lat1):
    ...     print('{0:6.2f}, {1:6.2f}'.format(x, y))
      9.00,  48.00
      9.00,  48.00
      9.00,  49.00
     10.49,  47.99
      9.00,  47.00
      7.51,  47.99

    The coordinates of the east and west directions won't come to lie on the
    latitude of the site because doesn't travel along the latitude circle but
    along a great circle.


    """

    # phi = 48.58611111 * pi/180.  # drs:  51.12527778 ; fbg: 47.87444444 ;
    # tur: 48.58611111 ; muc: 48.3372222
    # lon = 9.783888889 * pi/180.  # drs:  13.76972222 ; fbg: 8.005 ;
    # tur: 9.783888889 ; muc: 11.61277778
    phi = np.deg2rad(sitecoords[1])
    lam = np.deg2rad(sitecoords[0])

    a = np.deg2rad(-(180. + az))
    h = 0.5 * np.pi - r / re

    delta, tau = hor2aeq(a, h, phi)
    latc = np.rad2deg(delta)
    lonc = np.rad2deg(lam + tau)

    return lonc, latc


def __pol2lonlat(rng, az, sitecoords, re=6370040):
    """Alternative implementation using spherical geometry only.

    apparently it produces the same results as polar2lonlat.
    I wrote it because I suddenly doubted that the assumptions of the nautic
    triangle were wrong. I leave it here, in case someone might find it useful.

    Examples
    --------

    A few standard directions (North, South, North, East, South, West) with
    different distances (amounting to roughly 1°) from a site
    located at 48°N 9°E

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,]) * 1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> csite = (9.0, 48.0)
    >>> lon1, lat1= __pol2lonlat(r, az, csite)
    >>> for x, y in zip(lon1, lat1):
    ...     print('{0:6.2f}, {1:6.2f}'.format(x, y))
      9.00,  48.00
      9.00,  48.00
      9.00,  49.00
     10.49,  47.99
      9.00,  47.00
      7.51,  47.99

    The coordinates of the east and west directions won't come to lie on the
    latitude of the site because doesn't travel along the latitude circle but
    along a great circle.

    """
    phia = sitecoords[1]
    thea = sitecoords[0]

    polar_angle = np.deg2rad(90. - phia)
    r = rng / re

    easterly = az <= 180.
    # westerly = ~easterly
    a = np.deg2rad(np.where(easterly, az, az - 180.))

    m = np.arccos(np.cos(r) * np.cos(polar_angle) +
                  np.sin(r) * np.sin(polar_angle) * np.cos(a))
    g = np.arcsin((np.sin(r) * np.sin(a)) / (np.sin(m)))

    return thea + np.rad2deg(np.where(easterly, g, -g)), 90. - np.rad2deg(m)


def polar2lonlatalt(r, az, elev, sitecoords, re=6370040.):
    """Transforms polar coordinates to lon/lat/altitude coordinates.

    Explicitely accounts for the beam's elevation angle and for the altitude
    of the radar location.

    This is an alternative implementation based on VisAD code
    (see VisAD-Radar3DCoordinateSystem :cite:`VisAD-Radar3DCoordinateSystem`
    and VisAD-Home :cite:`VisAD-Home` ).

    VisAD code has been translated to Python from Java.

    Nomenclature tries to stick to VisAD code for the sake of comparibility,
    however, names of arguments are the same as for polar2lonlat...

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    az : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of three floats
        the lon / lat coordinates of the radar location and its altitude
        a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    re : float
        earth's radius [m]

    Returns
    -------
    output : tuple
        Tuple of three :class:`numpy:numpy.ndarray`
        (longitudes, latitudes, altitudes).

    Examples
    --------
    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])*1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> th = np.array([0.,   0.,   0.,   0.,   0.,  0.5,])
    >>> csite = (9.0, 48.0)
    >>> lon1, lat1, alt1 = polar2lonlatalt(r, az, th, csite)
    >>> for x, y, z in zip(lon1, lat1, alt1):
    ...     print('{0:7.4f}, {1:7.4f}, {2:7.4f}'.format(x, y, z))
    ...
     9.0000, 48.0000,  0.0000
     9.0000, 48.0000,  0.0000
     9.0000, 48.9983, 967.0320
    10.4919, 48.0000, 967.0320
     9.0000, 47.0017, 967.0320
     7.5084, 48.0000, 1935.4568

    """
    centlon = sitecoords[0]
    centlat = sitecoords[1]
    try:
        centalt = sitecoords[2]
    except Exception:
        centalt = 0.
    # local earth radius
    re = re + centalt

    cosaz = np.cos(np.deg2rad(az))
    sinaz = np.sin(np.deg2rad(az))
    # assume azimuth = 0 at north, then clockwise
    coselev = np.cos(np.deg2rad(elev))
    sinelev = np.sin(np.deg2rad(elev))
    rp = np.sqrt(re * re + r * r + 2.0 * sinelev * re * r)

    # altitudes
    alts = rp - re + centalt

    angle = np.arcsin(coselev * r / rp)  # really sin(elev+90)
    radp = re * angle
    lats = centlat + cosaz * radp / _latscale()
    lons = centlon + sinaz * radp / _lonscale(centlat)

    return lons, lats, alts


def polar2lonlatalt_n(r, az, elev, sitecoords, re=None, ke=4. / 3.):
    """Transforms polar coordinates (of a PPI) to longitude/latitude \
    coordinates taking elevation angle and refractivity into account.

    It takes the shortening of the great circle
    distance with increasing  elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    az : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0 and 360°.
        These are assumed to start with 0° pointing north and counted
        positive clockwise!
    th : scalar or :class:`numpy:numpy.ndarray` of the same shape as az
        Elevation angles in degrees starting with 0° at horizontal to 90°
        pointing vertically upwards from the radar
    sitecoords : a sequence of two or three floats
        The lon / lat coordinates of the radar location, the third value,
        if present, will be interpreted as the height of the site above the
        geoid (i.e. sphere)
    re : float
        Earth's radius [m], if None, `get_earth_radius` will be used to
        determine the equivalent radius of the WGS84 ellipsoid for the
        latitude given in sitecoords.
    ke : float
        Adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    lon, lat, alt : tuple of :class:`numpy:numpy.ndarray`
        Three arrays containing the spherical longitude and latitude
        coordinates as well as the altitude of the beam.

    Note
    ----
    The function uses osgeo/gdal functionality to reproject from azimuthal
    equidistant projection to spherical geographical coordinates.
    The earth model for this conversion is therefore spherical.
    This should not introduce too much error for common radar coverages, but
    you should be aware of this, when trying to do high resolution spatial
    analyses.

    Examples
    --------

    A few standard directions (North, South, North, East, South, West) with
    different distances (amounting to roughly 1°) from a site
    located at 48°N 9°E

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])*1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> th = np.array([0.,   0.,   0.,   0.,   0.,  0.5,])
    >>> csite = (9.0, 48.0)
    >>> lon1, lat1, alt1 = polar2lonlatalt_n(r, az, th, csite)
    >>> for x, y, z in zip(lon1, lat1, alt1):
    ...     print( '{0:7.4f}, {1:7.4f}, {2:7.4f}'.format(x, y, z))
    ...
     9.0000, 48.0000,  0.0000
     9.0000, 48.0000,  0.0000
     9.0000, 48.9989, 725.7160
    10.4927, 47.9903, 725.7160
     9.0000, 47.0011, 725.7160
     7.5076, 47.9903, 1694.2234

    Here, the coordinates of the east and west directions won't come to lie on
    the latitude of the site because the beam doesn't travel along the latitude
    circle but along a great circle.

    See :ref:`notebooks/basics/wradlib_workflow.ipynb#\
Georeferencing-and-Projection`.

    """
    # if site altitude is present, use it, else assume it to be zero
    try:
        centalt = sitecoords[2]
    except Exception:
        centalt = 0.

    # if no radius is given, get the approximate radius of the WGS84
    # ellipsoid for the site's latitude
    if re is None:
        re = get_earth_radius(sitecoords[1])

    # local earth radius
    re = re + centalt

    # altitude is calculated using the formulas of Doviak
    alt = beam_height_n(r, elev, re, ke) + centalt
    # same goes for the on-ground distance
    arc = arc_distance_n(r, elev, re, ke)

    # define the two projections
    # for the radar it's azimuthal equidistant projection
    rad = proj4_to_osr(('+proj=aeqd +lon_0={lon:f} +lat_0={lat:f} +a={re:f} ' +
                        '+b={re:f}').format(lon=sitecoords[0],
                                            lat=sitecoords[1],
                                            re=re))
    # for output we'd like to have spherical coordinates
    sph = proj4_to_osr('+proj=latlong +a={re:f} +b={re:f}'.format(re=re))

    # projected coordinates such as aeqd must be passed as x,y cartesian
    # coordinates and thus we have to convert the polar ones
    x = arc * np.cos(np.radians(90 - az))
    y = arc * np.sin(np.radians(90 - az))

    # then it's just a matter of invoking reproject
    lon, lat = reproject(x, y, projection_source=rad, projection_target=sph)

    return lon, lat, alt


def centroid2polyvert(centroid, delta):
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
               direction. If `delta` is scalar, it is assumed to apply to
               both dimensions.

    Returns
    -------
    vertices : :class:`numpy:numpy.ndarray`
               An array with 5 vertices per centroid.

    Note
    ----
    The function can currently only deal with 2-D data (If you come up with a
    higher dimensional version of 'clockwise' you're welcome to add it).
    The data is then assumed to be organized within the `centroid` array with
    the last dimension being the 2-D coordinates of each point.

    Examples
    --------

    >>> centroid2polyvert([0., 1.], [0.5, 1.5])
    array([[-0.5, -0.5],
           [-0.5,  2.5],
           [ 0.5,  2.5],
           [ 0.5, -0.5],
           [-0.5, -0.5]])
    >>> centroid2polyvert(np.arange(4).reshape((2,2)), 0.5)
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
    if (cent.shape[0] == 2) and (cent.shape[-1] != 2):
        cent = np.transpose(cent)
    assert cent.shape[-1] == 2
    dshape = [1] * cent.ndim
    dshape.insert(-1, 5)
    dshape[-1] = 2

    d = np.array([[-1., -1.],
                  [-1., 1.],
                  [1., 1.],
                  [1., -1.],
                  [-1., -1.]]).reshape(tuple(dshape))

    return np.asanyarray(centroid)[..., None, :] + d * np.asanyarray(delta)


def polar2polyvert(r, az, sitecoords):
    """
    Generate 2-D polygon vertices directly from polar coordinates.

    This is an alternative to :meth:`~wradlib.georef.centroid2polyvert` which
    does not use centroids, but generates the polygon vertices by simply
    connecting the corners of the radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    :meth:`~wradlib.georef.polar2lonlat`.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]; r defines the exterior boundaries of the range
        bins! (not the centroids). Thus, values must be positive!
    az : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main
        beam lobe!
        The first angle can start at any values, but make sure the array is
        sorted continuously positively clockwise and the angles are
        equidistant. An angle if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices in lon/lat with shape(num_vertices,
        num_vertex_nodes, 2). The last dimension carries the longitudes on
        the first position, the latitudes on the second (lon: output[:,:,0],
        lat: output[:,:,1]

    Examples
    --------
    >>> import wradlib.georef as georef  # noqa
    >>> import numpy as np
    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as pl
    >>> #pl.interactive(True)
    >>> # define the polar coordinates and the site coordinates in lat/lon
    >>> r = np.array([50., 100., 150., 200.])
    >>> # _check_polar_coords fails in next line
    >>> # az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])
    >>> az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
    >>> sitecoords = (9.0, 48.0)
    >>> polygons = georef.polar2polyvert(r, az, sitecoords)
    >>> # plot the resulting mesh
    >>> fig = pl.figure()
    >>> ax = fig.add_subplot(111)
    >>> #polycoll = mpl.collections.PolyCollection(vertices,closed=True, facecolors=None)  # noqa
    >>> polycoll = mpl.collections.PolyCollection(polygons,closed=True, facecolors='None')  # noqa
    >>> ret = ax.add_collection(polycoll, autolim=True)
    >>> pl.autoscale()
    >>> pl.show()

    """
    # prepare the range and azimuth array so they describe the boundaries of
    # a bin, not the centroid
    r, az = _check_polar_coords(r, az)
    r = np.insert(r, 0, r[0] - _get_range_resolution(r))
    az = az - 0.5 * _get_azimuth_resolution(az)
    az = np.append(az, az[0])
    az = np.where(az < 0, az + 360., az)

    # generate a grid of polar coordinates of bin corners
    r, az = np.meshgrid(r, az)
    # convert polar coordinates to lat/lon
    lon, lat = polar2lonlat(r, az, sitecoords)

    llc = np.transpose(np.vstack((lon[:-1, :-1].ravel(),
                                  lat[:-1, :-1].ravel())))
    ulc = np.transpose(np.vstack((lon[:-1, 1:].ravel(),
                                  lat[:-1, 1:].ravel())))
    urc = np.transpose(np.vstack((lon[1:, 1:].ravel(),
                                  lat[1:, 1:].ravel())))
    lrc = np.transpose(np.vstack((lon[1:, :-1].ravel(),
                                  lat[1:, :-1].ravel())))

    vertices = np.concatenate((llc, ulc, urc, lrc, llc)).reshape((-1, 5, 2),
                                                                 order='F')

    return vertices


def polar2centroids(r=None, az=None, sitecoords=None, range_res=None):
    """
    Computes the lat/lon centroids of the radar bins from the polar
    coordinates.

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
    az : :class:`numpy:numpy.ndarray`
        Array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main
        beam lobe!
        The first angle can start at any values, but make sure the array is
        sorted continuously positively clockwise and the angles are
        equidistant. An angle if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location
    range_res : float
        range resolution of radar measurement [m] in case it cannot be derived
        from r (single entry in r-array)

    Returns
    -------
    output : tuple
        Tuple of 2 :class:`numpy:numpy.ndarray` which describe the bin
        centroids longitude and latitude.

    Note
    ----
    Azimuth angles of 360 deg are internally converted to 0 deg.

    """
    # make sure the range and azimuth angles have the right properties
    r, az = _check_polar_coords(r, az)

    # to get the centroid, we only have to move the exterior bin boundaries by
    # half the resolution
    if range_res:
        r = r - 0.5 * range_res
    else:
        r = r - 0.5 * _get_range_resolution(r)
    # generate a polar grid and convert to lat/lon
    r, az = np.meshgrid(r, az)
    lon, lat = polar2lonlat(r, az, sitecoords)

    return lon, lat


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


def projected_bincoords_from_radarspecs(r, az, sitecoords, proj,
                                        range_res=None):
    """
    Convenience function to compute projected bin coordinates directly from
    radar site coordinates and range/azimuth specs

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]; r defines the exterior boundaries of the range
        bins! (not the centroids). Thus, values must be positive!
    az : :class:`numpy:numpy.ndarray`
    sitecoords : tuple
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main
        beam lobe! The first angle can start at any values, but make sure the
        array is sorted continuously positively clockwise and the angles are
        equidistant. An angle if 0 degree is pointing north.
    proj : osr.SpatialReference
        GDAL OSR Spatial Reference Object describing projection
    range_res : float
        range resolution of radar measurement [m] in case it cannot be derived
        from r (single entry in r-array)

    """
    cent_lon, cent_lat = polar2centroids(r, az, sitecoords,
                                         range_res=range_res)
    x, y = reproject(cent_lon, cent_lat, projection_target=proj)
    return x.ravel(), y.ravel()


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
