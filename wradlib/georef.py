#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Georeferencing
^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   beam_height_n
   arc_distance_n
   polar2lonlat
   polar2lonlatalt
   polar2lonlatalt_n
   polar2centroids
   polar2polyvert
   centroid2polyvert
   reproject
   create_osr
   proj4_to_osr
   epsg_to_osr
   wkt_to_osr
   projected_bincoords_from_radarspecs
   sweep_centroids
   read_gdal_values
   read_gdal_projection
   read_gdal_coordinates
   pixel_to_map3d
   pixel_to_map
   pixel_coordinates
   get_earth_radius
   get_radolan_grid
   reproject_raster_dataset
   resample_raster_dataset
   get_shape_coordinates
   correct_parallax
   sat2pol
   dist_from_orbit
   create_raster_dataset
   set_raster_origin
   extract_raster_dataset


"""

# Seitenlänge Zenit - Himmelsnordpol: 90°-phi
# Seitenlänge Himmelsnordpol - Gestirn: 90°-delta
# Seitenlänge Zenit - Gestirn: 90°-h
# Winkel Himmelsnordpol - Zenit - Gestirn: 180°-a
# Winkel Zenit - Himmelsnordpol - Gestirn: tau

# alpha - rektaszension
# delta - deklination
# theta - sternzeit
# tau = theta - alpha - stundenwinkel
# a - azimuth (von süden aus gezählt)
# h - Höhe über Horizont

from osgeo import gdal, osr, gdal_array
import numpy as np
from sys import exit
import warnings

from . import util as util


def hor2aeq(a, h, phi):
    """"""
    delta = np.arcsin(np.sin(h) * np.sin(phi) - np.cos(h) *
                      np.cos(a) * np.cos(phi))
    tau = np.arcsin(np.cos(h) * np.sin(a) / np.cos(delta))
    return delta, tau


def aeq2hor(tau, delta, phi):
    """"""
    h = np.arcsin(np.cos(delta) * np.cos(tau) * np.cos(phi) +
                  np.sin(delta) * np.sin(phi))
    a = np.arcsin(np.cos(delta) * np.sin(tau) / np.cos(h))
    return a, h


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

    l = np.deg2rad(90. - phia)
    r = rng / re

    easterly = az <= 180.
    # westerly = ~easterly
    a = np.deg2rad(np.where(easterly, az, az - 180.))

    m = np.arccos(np.cos(r) * np.cos(l) + np.sin(r) * np.sin(l) * np.cos(a))
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
    except:
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


def _latscale(re=6370040.):
    """Return meters per degree latitude assuming spherical earth
    """
    return 2 * np.pi * re / 360.


def _lonscale(lat, re=6370040.):
    """Return meters per degree longitude assuming spherical earth
    """
    return (2 * np.pi * re / 360.) * np.cos(np.deg2rad(lat))


def beam_height_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the height of a radar beam taking the refractivity of the
    atmosphere into account.

    Based on :cite:`Doviak1993` the beam height is calculated as

    .. math::

        h = \sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \sin\theta} - k_e r_e


    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    height : float
        height of the beam in [m]

    """
    return np.sqrt(r ** 2 + (ke * re) ** 2 +
                   2 * r * ke * re * np.sin(np.radians(theta))) - ke * re


def arc_distance_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the great circle distance of a radar beam over a sphere,
    taking the refractivity of the atmosphere into account.

    Based on :cite:`Doviak1993` the arc distance may be calculated as

    .. math::

        s = k_e r_e \arcsin\left(
        \frac{r \cos\theta}{k_e r_e + h_n(r, \theta, r_e, k_e)}\right)

    where :math:`h_n` would be provided by
    :meth:`~wradlib.georef.beam_height_n`

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : float
        great circle arc distance [m]

    See Also
    --------
    beam_height_n

    """
    return ke * re * np.arcsin((r * np.cos(np.radians(theta))) /
                               (ke * re + beam_height_n(r, theta, re, ke)))


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
    except:
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


def create_osr(projname, **kwargs):
    """Conveniently supports the construction of osr spatial reference objects

    .. versionadded:: 0.6.0

    Currently, the following projection names (argument *projname*)
    are supported:

    **"aeqd": Azimuthal Equidistant**

    needs the following keyword arguments:

        - *lat_0* (latitude at projection center),
        - *lon_0* (longitude at projection center),
        - *x_0* (false Easting, also known as x-offset),
        - *y_0* (false Northing, also known as y-offset)

    **"dwd-radolan" : RADOLAN Composite Coordinate System**

        - no additional arguments needed.

    Polar stereographic projection used by the German Weather Service (DWD)
    for all Radar composite products. See the final report on the RADOLAN
    project :cite:`DWD2004` for details.

    Parameters
    ----------
    projname : string
        "aeqd" or "dwd-radolan"
    kwargs : depends on projname - see above!

    Returns
    -------
    output : osr.SpatialReference
        GDAL/OSR object defining projection

    Examples
    --------

    See :ref:`notebooks/basics/wradlib_workflow.ipynb#\
Georeferencing-and-Projection`.

    """

    aeqd_wkt = ('PROJCS["unnamed",'
                'GEOGCS["WGS 84",'
                'DATUM["unknown",'
                'SPHEROID["WGS84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                'PROJECTION["Azimuthal_Equidistant"],'
                'PARAMETER["latitude_of_center", {0:-f}],'
                'PARAMETER["longitude_of_center", {1:-f}],'
                'PARAMETER["false_easting", {2:-f}],'
                'PARAMETER["false_northing", {3:-f}]]')

    radolan_wkt = ('PROJCS["Radolan projection",'
                   'GEOGCS["Radolan Coordinate System",'
                   'DATUM["Radolan Kugel",'
                   'SPHEROID["Erdkugel", 6370040.0, 0.0]],'
                   'PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],'
                   'UNIT["degree", 0.017453292519943295],'
                   'AXIS["Longitude", EAST],'
                   'AXIS["Latitude", NORTH]],'
                   'PROJECTION["polar_stereographic"],'
                   'PARAMETER["central_meridian", 10.0],'
                   'PARAMETER["latitude_of_origin", 60.0],'
                   'PARAMETER["scale_factor", {0:8.10f}],'
                   'PARAMETER["false_easting", 0.0],'
                   'PARAMETER["false_northing", 0.0],'
                   'UNIT["m*1000.0", 1000.0],'
                   'AXIS["X", EAST],'
                   'AXIS["Y", NORTH]]')
    #                  'AUTHORITY["USER","100000"]]'

    proj = osr.SpatialReference()

    if projname == "aeqd":
        # Azimuthal Equidistant
        if "x_0" in kwargs:
            proj.ImportFromWkt(aeqd_wkt.format(kwargs["lat_0"],
                                               kwargs["lon_0"],
                                               kwargs["x_0"],
                                               kwargs["y_0"]))
        else:
            proj.ImportFromWkt(aeqd_wkt.format(kwargs["lat_0"],
                                               kwargs["lon_0"], 0, 0))

    elif projname == "dwd-radolan":
        # DWD-RADOLAN polar stereographic projection
        scale = (1. + np.sin(np.radians(60.))) / (1. + np.sin(np.radians(90.)))
        proj.ImportFromWkt(radolan_wkt.format(scale))

    else:
        print("No convenience support for projection %r, yet." % projname)
        print("You need to create projection by using other means...")
        exit(1)

    return proj


def proj4_to_osr(proj4str):
    """Transform a proj4 string to an osr spatial reference object

    Parameters
    ----------
    proj4str : string
        Proj4 string describing projection

    Examples
    --------

    See :ref:`notebooks/radolan/radolan_grid.ipynb#PROJ.4`.

    """
    proj = None
    if proj4str:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(proj4str)
    else:
        proj = get_default_projection()
    return proj


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


def get_earth_radius(latitude, sr=None):
    r"""
    Get the radius of the Earth (in km) for a given Spheroid model (sr) at a
    given position

    .. math::

        R^2 = \frac{a^4 \cos(f)^2 + b^4 \sin(f)^2}
        {a^2 \cos(f)^2 + b^2 \sin(f)^2}

    Parameters
    ----------
    sr : osr object
        spatial reference;
    latitude : float
        geodetic latitude in degrees;

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if sr is None:
        sr = get_default_projection()
    RADIUS_E = sr.GetSemiMajor()
    RADIUS_P = sr.GetSemiMinor()
    latitude = np.radians(latitude)
    radius = np.sqrt((np.power(RADIUS_E, 4) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 4) * np.power(np.sin(latitude), 2)) /
                     (np.power(RADIUS_E, 2) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 2) * np.power(np.sin(latitude), 2)))
    return (radius)


def pixel_coordinates(nx, ny, mode="centers"):
    """Get pixel coordinates from a regular grid with dimension nx by ny.

    Parameters
    ----------
    nx : int
        xsize
    ny : int
        ysize
    mode : string
        `centers` or `centroids` to return the pixel centers coordinates
        otherwise the pixel edges coordinates will be returned
    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
         Array of shape (ny,nx) with pixel coordinates (x,y)

    """
    if mode == "centroids":
        mode = "centers"
    x = np.linspace(0, nx, num=nx + 1)
    y = np.linspace(0, ny, num=ny + 1)
    if mode == "centers":
        x = x + 0.5
        y = y + 0.5
        x = np.delete(x, -1)
        y = np.delete(y, -1)
    X, Y = np.meshgrid(x, y)
    coordinates = np.empty(X.shape + (2,))
    coordinates[:, :, 0] = X
    coordinates[:, :, 1] = Y
    return (coordinates)


def pixel_to_map(geotransform, coordinates):
    """Apply a geographical transformation to return map coordinates from
    pixel coordinates.

    Parameters
    ----------
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector:

            - geotransform[0] = East/West location of Upper Left corner
            - geotransform[1] = X pixel size
            - geotransform[2] = X pixel rotation
            - geotransform[3] = North/South location of Upper Left corner
            - geotransform[4] = Y pixel rotation
            - geotransform[5] = Y pixel size
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        3d array with map coordinates x,y
    """
    coordinates_map = np.empty(coordinates.shape)
    coordinates_map[..., 0] = (geotransform[0] +
                               geotransform[1] * coordinates[..., 0] +
                               geotransform[2] * coordinates[..., 1])
    coordinates_map[..., 1] = (geotransform[3] +
                               geotransform[4] * coordinates[..., 0] +
                               geotransform[5] * coordinates[..., 1])
    return (coordinates_map)


def pixel_to_map3d(geotransform, coordinates, z=None):
    """Apply a geographical transformation to return 3D map coordinates from
    pixel coordinates.

    Parameters
    ----------
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector
        (see :meth:`~wradlib.georef.pixel_to_map`)
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates;
    z : string
        method to compute the z coordinates (height above ellipsoid):

            - None : default, z equals zero
            - srtm : not available yet

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        4d array with map coordinates x,y,z

    """

    coordinates_map = np.empty(coordinates.shape[:-1] + (3,))
    coordinates_map[..., 0:2] = pixel_to_map(geotransform, coordinates)
    coordinates_map[..., 2] = np.zeros(coordinates.shape[:-1])
    return (coordinates_map)


def read_gdal_coordinates(dataset, mode='centers', z=True):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing
    mode : string
        either 'centers' or 'borders'
    z : boolean
        True to get height coordinates (zero).

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
        Array of projected coordinates (x,y,z)

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    coordinates_pixel = pixel_coordinates(dataset.RasterXSize,
                                          dataset.RasterYSize, mode)
    geotransform = dataset.GetGeoTransform()
    if z:
        coordinates = pixel_to_map3d(geotransform, coordinates_pixel)
    else:
        coordinates = pixel_to_map(geotransform, coordinates_pixel)
    return (coordinates)


def read_gdal_projection(dset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dset : gdal.Dataset

    Returns
    -------
    srs : OSR.SpatialReference
        dataset projection object

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    wkt = dset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    # src = None
    return srs


def read_gdal_values(dataset=None, nodata=None):
    """Read values from a gdal object.

    Parameters
    ----------
    data : gdal object
    nodata : boolean
        option to deal with nodata values replacing it with nans.

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    nbands = dataset.RasterCount

    # data values
    bands = []
    for i in range(nbands):
        band = dataset.GetRasterBand(i + 1)
        nd = band.GetNoDataValue()
        data = band.ReadAsArray()
        if nodata is not None:
            data[data == nd] = nodata
        bands.append(data)

    return np.squeeze(np.stack(bands))


def reproject(*args, **kwargs):
    """Transform coordinates from a source projection to a target projection.

    Call signatures::

        reproject(C, **kwargs)
        reproject(X, Y, **kwargs)
        reproject(X, Y, Z, **kwargs)

    *C* is the np array of source coordinates.
    *X*, *Y* and *Z* specify arrays of x, y, and z coordinate values

    Parameters
    ----------
    C : multidimensional :class:`numpy:numpy.ndarray`
        Array of shape (...,2) or (...,3) with coordinates (x,y) or (x,y,z)
        respectively
    X : :class:`numpy:numpy.ndarray`
        Array of x coordinates
    Y : :class:`numpy:numpy.ndarray`
        Array of y coordinates
    Z : :class:`numpy:numpy.ndarray`
        Array of z coordinates

    Keyword Arguments
    -----------------
    projection_source : osr object
        defaults to EPSG(4326)
    projection_target : osr object
        defaults to EPSG(4326)

    Returns
    -------
    trans : :class:`numpy:numpy.ndarray`
        Array of reprojected coordinates x,y (...,2) or x,y,z (...,3)
        depending on input array.
    X, Y : :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y coordinates, shape depending on input array
    X, Y, Z: :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y,z coordinates, shape depending on input array

    Examples
    --------

    See :ref:`notebooks/georeferencing/wradlib_georef_example.ipynb`.

    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape = C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1, numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError('Input Array column mismatch '
                            'to %s' % ('reproject'))
    else:
        if len(args) == 2:
            X, Y = (np.asanyarray(arg) for arg in args)
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            numCols = 3
        else:
            raise TypeError('Illegal arguments to %s' % ('reproject'))

        xshape = X.shape
        yshape = Y.shape

        if xshape != yshape:
            raise TypeError('Incompatible X, Y inputs to %s' % ('reproject'))

        if 'Z' in locals():
            if xshape != zshape:
                raise TypeError('Incompatible Z input to %s' % ('reproject'))
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None],
                                Z.ravel()[:, None]], axis=1)
        else:
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None]], axis=1)

    projection_source = kwargs.get('projection_source',
                                   get_default_projection())
    projection_target = kwargs.get('projection_target',
                                   get_default_projection())

    ct = osr.CoordinateTransformation(projection_source, projection_target)
    trans = np.array(ct.TransformPoints(C))

    if len(args) == 1:
        # here we could do this one
        # return(np.array(ct.TransformPoints(C))[...,0:numCols]))
        # or this one
        trans = trans[:, 0:numCols].reshape(cshape)
        return trans
    else:
        X = trans[:, 0].reshape(xshape)
        Y = trans[:, 1].reshape(yshape)
        if len(args) == 2:
            return X, Y
        if len(args) == 3:
            Z = trans[:, 2].reshape(zshape)
            return X, Y, Z


def get_default_projection():
    """Create a default projection object (wgs84)"""
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    return proj


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


def epsg_to_osr(epsg=None):
    """Create osr spatial reference object from EPSG number

    .. versionadded:: 0.6.0

    Parameters
    ----------
    epsg : int
        EPSG-Number defining the coordinate system

    Returns
    -------
    proj : osr.SpatialReference
        GDAL/OSR object defining projection
    """
    proj = None
    if epsg:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg)
    else:
        proj = get_default_projection()
    return proj


def wkt_to_osr(wkt=None):
    """Create osr spatial reference object from WKT string

    .. versionadded:: 0.6.0

    Parameters
    ----------
    wkt : str
        WTK string defining the coordinate reference system

    Returns
    -------
    proj : osr.SpatialReference
        GDAL/OSR object defining projection

    """
    proj = None
    if wkt:
        proj = osr.SpatialReference()
        proj.ImportFromWkt(wkt)
    else:
        proj = get_default_projection()
    return proj


def get_radolan_coords(lon, lat, trig=False):
    """
    Calculates x,y coordinates of radolan grid from lon, lat

    .. versionadded:: 0.4.0

    Parameters
    ----------

    lon :   float, :class:`numpy:numpy.ndarray` of floats
        longitude
    lat :   float, :class:`numpy:numpy.ndarray` of floats
        latitude
    trig : boolean
        if True, uses trigonometric formulas for calculation,
        otherwise osr transformations
        if False, uses osr spatial reference system to transform
        between projections
        `trig` is recommended to be False, however, the two ways of
        computation are expected to be equivalent.
    """

    if trig:
        # calculation of x_0 and y_0 coordinates of radolan grid
        # as described in the format description
        phi_0 = np.radians(60)
        phi_m = np.radians(lat)
        lam_0 = 10
        lam_m = lon
        lam = np.radians(lam_m - lam_0)
        er = 6370.040
        m_phi = (1 + np.sin(phi_0)) / (1 + np.sin(phi_m))
        x = er * m_phi * np.cos(phi_m) * np.sin(lam)
        y = - er * m_phi * np.cos(phi_m) * np.cos(lam)
    else:
        # create radolan projection osr object
        proj_stereo = create_osr("dwd-radolan")

        # create wgs84 projection osr object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        x, y = reproject(lon, lat, projection_source=proj_wgs,
                         projection_target=proj_stereo)

    return x, y


def get_radolan_grid(nrows=None, ncols=None, trig=False, wgs84=False):
    """Calculates x/y coordinates of radolan grid of the German Weather Service

    .. versionadded:: 0.4.0

    Returns the x,y coordinates of the radolan grid positions
    (lower left corner of every pixel). The radolan grid is a
    polarstereographic projection, the projection information was taken from
    RADOLAN-RADVOR-OP Kompositformat_2.2.2  :cite:`DWD2009`

    .. table:: Coordinates for 900km x 900km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  3.5889E  |  46.9526N  | -523.4622 | -4658.645 |
        +------------+-----------+------------+-----------+-----------+
        | LowerRight | 14.6209E  |  47.0705N  |  376.5378 | -4658.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperRight | 15.7208E  |  54.7405N  |  376.5378 | -3758.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperLeft  |  2.0715E  |  54.5877N  | -523.4622 | -3758.645 |
        +------------+-----------+------------+-----------+-----------+

    .. table:: Coordinates for 1100km x 900km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  4.6759E  |  46.1929N  | -443.4622 | -4758.645 |
        +------------+-----------+------------+-----------+-----------+
        | LowerRight | 15.4801E  |  46.1827N  |  456.5378 | -4758.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperRight | 17.1128E  |  55.5342N  |  456.5378 | -3658.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperLeft  |  3.0889E  |  55.5482N  | -433.4622 | -3658.645 |
        +------------+-----------+------------+-----------+-----------+

    .. table:: Coordinates for 1500km x 1400km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  2.3419E  |  43.9336N  | -673.4622 | -5008.645 |
        +------------+-----------+------------+-----------+-----------+

    Parameters
    ----------
    nrows : int
        number of rows (460, 900 by default, 1100, 1500)
    ncols : int
        number of columns (460, 900 by default, 1400)
    trig : boolean
        if True, uses trigonometric formulas for calculation
        if False, uses osr spatial reference system to transform between
        projections
        `trig` is recommended to be False, however, the two ways of computation
        are expected to be equivalent.
    wgs84 : boolean
        if True, output coordinates are in wgs84 lonlat format (default: False)

    Returns
    -------
    radolan_grid : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) xy- or lonlat-grid.

    Examples
    --------

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef  # noqa
    >>> radolan_grid = georef.get_radolan_grid()
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (-523.4622, -4658.6447)

    >>> # using pure trigonometric transformations
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(trig=True)
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (-523.4622, -4658.6447)

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(1500, 1400)
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (1500, 1400, 2), (-673.4622, -5008.6447)

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(900, 900, wgs84=True)
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (3.5889, 46.9526)

    See :ref:`notebooks/radolan/radolan_grid.ipynb#Polar-Stereographic-Projection`.  # noqa

    Raises
    ------
        TypeError, ValueError
    """

    # setup default parameters in dicts
    tiny = {'j_0': 450, 'i_0': 450, 'res': 2}
    small = {'j_0': 460, 'i_0': 460, 'res': 2}
    normal = {'j_0': 450, 'i_0': 450, 'res': 1}
    normal_wx = {'j_0': 370, 'i_0': 550, 'res': 1}
    extended = {'j_0': 600, 'i_0': 800, 'res': 1}
    griddefs = {(450, 450): tiny, (460, 460): small,
                (900, 900): normal, (1100, 900): normal_wx,
                (1500, 1400): extended}

    # type and value checking
    if nrows and ncols:
        if not (isinstance(nrows, int) and isinstance(ncols, int)):
            raise TypeError("wradlib.georef: Parameter *nrows* "
                            "and *ncols* not integer")
        if (nrows, ncols) not in griddefs.keys():
            raise ValueError("wradlib.georef: Parameter *nrows* "
                             "and *ncols* mismatch.")
    else:
        # fallback for call without parameters
        nrows = 900
        ncols = 900

    # tiny, small, normal or extended grid check
    # reference point changes according to radolan composit format
    j_0 = griddefs[(nrows, ncols)]['j_0']
    i_0 = griddefs[(nrows, ncols)]['i_0']
    res = griddefs[(nrows, ncols)]['res']

    x_0, y_0 = get_radolan_coords(9.0, 51.0, trig=trig)

    x_arr = np.arange(x_0 - j_0, x_0 - j_0 + ncols * res, res)
    y_arr = np.arange(y_0 - i_0, y_0 - i_0 + nrows * res, res)
    x, y = np.meshgrid(x_arr, y_arr)

    radolan_grid = np.dstack((x, y))

    if wgs84:

        if trig:
            # inverse projection
            lon0 = 10.  # central meridian of projection
            lat0 = 60.  # standard parallel of projection

            sinlat0 = np.sin(np.radians(lat0))

            fac = (6370.040 ** 2.) * ((1. + sinlat0) ** 2.)
            lon = np.degrees(np.arctan((-x / y))) + lon0
            lat = np.degrees(np.arcsin((fac - (x ** 2. + y ** 2.)) /
                                       (fac + (x ** 2. + y ** 2.))))
            radolan_grid = np.dstack((lon, lat))
        else:
            # create radolan projection osr object
            proj_stereo = create_osr("dwd-radolan")

            # create wgs84 projection osr object
            proj_wgs = osr.SpatialReference()
            proj_wgs.ImportFromEPSG(4326)

            radolan_grid = reproject(radolan_grid,
                                     projection_source=proj_stereo,
                                     projection_target=proj_wgs)

    return radolan_grid


def reproject_raster_dataset(src_ds, **kwargs):
    """Reproject/Resample given dataset according to keyword arguments

    .. versionadded:: 0.10.0

    # function inspired from github project
    # https://github.com/profLewis/geogg122

    Parameters
    ----------
    src_ds : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)
    spacing : float
        float or tuple of two floats
        pixel spacing of destination dataset, same unit as pixel coordinates
    size : int
        tuple of two ints
        X/YRasterSize of destination dataset
    resample : GDALResampleAlg
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    projection_target : osr object
        destination dataset projection, defaults to None
    align : bool or Point
        If False, there is no destination grid aligment.
        If True, aligns the destination grid to the next integer multiple of
        destination grid.
        If Point (tuple, list of upper-left x,y-coordinate), the destination
        grid is aligned to this point.

    Returns
    -------
    dst_ds : gdal.Dataset
        reprojected/resampled raster dataset
    """

    # checking kwargs
    spacing = kwargs.pop('spacing', None)
    size = kwargs.pop('size', None)
    resample = kwargs.pop('resample', gdal.GRA_Bilinear)
    src_srs = kwargs.pop('projection_source', None)
    dst_srs = kwargs.pop('projection_target', None)
    align = kwargs.pop('align', False)

    # Get the GeoTransform vector
    src_geo = src_ds.GetGeoTransform()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    # get extent
    ulx = src_geo[0]
    uly = src_geo[3]
    lrx = src_geo[0] + src_geo[1] * x_size
    lry = src_geo[3] + src_geo[5] * y_size

    extent = np.array([[[ulx, uly],
                        [lrx, uly]],
                       [[ulx, lry],
                        [lrx, lry]]])

    if dst_srs:
        print("dest_src available")
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_ds.GetProjection())

        # Transformation
        extent = reproject(extent, projection_source=src_srs,
                           projection_target=dst_srs)

        # wkt needed
        src_srs = src_srs.ExportToWkt()
        dst_srs = dst_srs.ExportToWkt()

    (ulx, uly, urx, ury,
     llx, lly, lrx, lry) = tuple(list(extent.flatten().tolist()))

    # align grid to destination raster or UL-corner point
    if align:
        try:
            ulx, uly = align
        except TypeError:
            pass

        ulx = int(max(np.floor(ulx), np.floor(llx)))
        uly = int(min(np.ceil(uly), np.ceil(ury)))
        lrx = int(min(np.ceil(lrx), np.ceil(urx)))
        lry = int(max(np.floor(lry), np.floor(lly)))

    # calculate cols/rows or xspacing/yspacing
    if spacing:
        try:
            x_ps, y_ps = spacing
        except TypeError:
            x_ps = spacing
            y_ps = spacing

        cols = int(abs(lrx - ulx) / x_ps)
        rows = int(abs(uly - lry) / y_ps)
    elif size:
        cols, rows = size
        x_ps = x_size * src_geo[1] / cols
        y_ps = y_size * abs(src_geo[5]) / rows
    else:
        raise NameError("Whether keyword 'spacing' or 'size' must be given")

    # create destination in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    print(rows, cols)
    # and set RasterSize according ro cols/rows
    dst_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Float32)

    # Create the destination GeoTransform with changed x/y spacing
    dst_geo = (ulx, x_ps, src_geo[2], uly, src_geo[4], -y_ps)

    # apply GeoTransform to destination dataset
    dst_ds.SetGeoTransform(dst_geo)

    # nodata handling, need to initialize dst_ds with nodata
    src_band = src_ds.GetRasterBand(1)
    nodata = src_band.GetNoDataValue()
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(nodata)
    dst_band.WriteArray(np.ones((rows, cols)) * nodata)
    dst_band.FlushCache()

    # resample and reproject dataset
    gdal.ReprojectImage(src_ds, dst_ds, src_srs, dst_srs, resample)

    return dst_ds


@util.deprecated(reproject_raster_dataset)
def resample_raster_dataset(src_ds, **kwargs):
    """Resample given dataset according to keyword arguments

    .. versionadded:: 0.6.0

    # function inspired from github project
    # https://github.com/profLewis/geogg122

    Parameters
    ----------
    src_ds : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)
    spacing : float
        float or tuple of two floats
        pixel spacing of resampled dataset, same unit as pixel coordinates
    size : int
        tuple of two ints
        X/YRasterSize of resampled dataset
    resample : GDALResampleAlg
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12

    Returns
    -------
    dst_ds : gdal.Dataset
        resampled raster dataset
    """

    # checking kwargs
    spacing = kwargs.pop('spacing', None)
    size = kwargs.pop('size', None)
    resample = kwargs.pop('resample', gdal.GRA_Bilinear)

    # Get the GeoTransform vector
    src_geo = src_ds.GetGeoTransform()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    # calculate cols/rows or xspacing/yspacing
    if spacing:
        try:
            x_ps, y_ps = spacing
        except TypeError:
            x_ps = spacing
            y_ps = spacing
        cols = int((x_size * src_geo[1]) / x_ps)
        rows = int((y_size * abs(src_geo[5])) / y_ps)
    elif size:
        cols, rows = size
        x_ps = x_size * src_geo[1] / cols
        y_ps = y_size * abs(src_geo[5]) / rows
    else:
        print("whether keyword 'spacing' or 'size' must be given")
        raise

    # create destination in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    # and set RasterSize according cols/rows
    dst_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Float32)

    # Create the destination GeoTransform with changed x/y spacing
    dst_geo = (src_geo[0], x_ps, src_geo[2], src_geo[3], src_geo[4], -y_ps)

    # apply GeoTransform to destination dataset
    dst_ds.SetGeoTransform(dst_geo)

    # resample dataset
    gdal.ReprojectImage(src_ds, dst_ds, None, None, resample)

    return dst_ds


def get_shape_points(geom):
    """
    Extract coordinate points from given ogr geometry as generator object

    If geometries are nested, function recurses.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    geom : ogr.Geometry

    Returns
    -------
    result : generator object
        expands to Nx2 dimensional nested point arrays
    """

    type = geom.GetGeometryType()
    if type:
        # 1D Geometries, LINESTRINGS
        if type == 2:
            result = np.array(geom.GetPoints())
            yield result
        # RINGS, POLYGONS, MULTIPOLYGONS, MULTILINESTRINGS
        elif type > 2:
            # iterate over geometries and recurse
            for item in geom:
                for result in get_shape_points(item):
                    yield result
    else:
        print("Unknown Geometry")


def transform_geometry(geom, dest_srs):
    """
    Perform geotransformation to given destination SpatialReferenceSystem

    It transforms coordinates to a given destination osr spatial reference
    if a geotransform is neccessary.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    geom : ogr.geometry
    dest_srs: osr.SpatialReference
        Destination Projection

    Returns
    -------
    geom : ogr.Geometry
        Transformed Geometry
    """

    # transform if not the same spatial reference system
    if not geom.GetSpatialReference().IsSame(dest_srs):
        geom.TransformTo(dest_srs)

    return geom


def get_shape_coordinates(layer, **kwargs):
    """
    Function iterates over gdal ogr layer features and packs extracted shape
    coordinate points into nested ndarray

    It transforms coordinates to a given destination osr spatial reference if
    dest_srs is given and a geotransform is neccessary.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    layer : ogr.Layer

    Keywords
    --------
    dest_srs: osr.SpatialReference
        Destination Projection
    key : string
        attribute key to extract from layer feature

    Returns
    -------
    shp : nested :class:`numpy:numpy.ndarray`
        Dimension of subarrays Nx2
        extracted shape coordinate points
    attrs : list
        List of attributes extracted from features
    """

    shp = []

    dest_srs = kwargs.get('dest_srs', None)
    key = kwargs.get('key', None)
    if key:
        attrs = []
    else:
        attrs = None

    for i in range(layer.GetFeatureCount()):
        feature = layer.GetNextFeature()
        if feature:
            if key:
                attrs.append(feature[key])
            geom = feature.GetGeometryRef()
            if dest_srs:
                transform_geometry(geom, dest_srs)
            # get list of xy-coordinates
            reslist = list(get_shape_points(geom))
            shp.append(np.squeeze(np.array(reslist)))

    shp = np.squeeze(np.array(shp))

    return shp, attrs


def correct_parallax(pr_xy, nbin, drt, alpha):
    """Adjust the geo-locations of the PR pixels

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    The `pr_xy` coordinates of the PR beam footprints need to be in the
    azimuthal equidistant projection of the ground radar. This ensures that the
    ground radar is fixed at xy-coordinate (0, 0), and every PR bin has its
    relative xy-coordinates with respect to the ground radar site.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_xy : :class:`numpy:numpy.ndarray`
        Array of xy-coordinates of shape (nscans, nbeams, 2)
    nbin : int
        Number of bins along PR beam.
    drt : float
        Gate lenght of PR in meter.
    alpha: :class:`numpy:numpy.ndarray`
        Array of depression angles of the PR beams with shape (nbeams).

    Returns
    -------

    pr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins, 2).
    r_pr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to PR platform of shape (nbins).
    z_pr : :class:`numpy:numpy.ndarray`
        Array of PR bin altitudes of shape (nbeams, nbins).
    """
    # get x,y-grids
    pr_x = pr_xy[..., 0]
    pr_y = pr_xy[..., 1]

    # create range array from ground to sat
    r_pr_inv = np.arange(nbin) * drt

    # calculate height of bin
    z_pr = r_pr_inv * np.cos(np.deg2rad(alpha))[:, np.newaxis]
    # calculate bin ground xy-displacement length
    ds = r_pr_inv * np.sin(np.deg2rad(alpha))[:, np.newaxis]

    # calculate x,y-differences between ground coordinate
    # and center ground coordinate [25th element]
    center = int(np.floor(len(pr_x[-1]) / 2.))
    xdiff = pr_x[:, center][:, np.newaxis] - pr_x
    ydiff = pr_y[:, center][:, np.newaxis] - pr_y

    # assuming ydiff and xdiff being a triangles adjacent and
    # opposite this calculates the xy-angle of the PR scan
    ang = np.arctan2(ydiff, xdiff)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.cos(ang)[..., np.newaxis]
    dy = ds * np.sin(ang)[..., np.newaxis]

    # add displacement to PR ground coordinates
    pr_xp = dx + pr_x[..., np.newaxis]
    pr_yp = dy + pr_y[..., np.newaxis]

    return np.stack((pr_xp, pr_yp), axis=3), r_pr_inv, z_pr


def sat2pol(pr_xyz, gr_site_alt, re):
    """Returns spherical coordinates of PR bins as seen from the GR location.

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM. With *GR*, we refer to terrestrial weather radars
    (ground radars).

    For this function to work, the `pr_xyz` coordinates of the PR bins need
    to be in the azimuthal equidistant projection of the ground radar! This
    ensures that the ground radar is fixed at xy-coordinate (0, 0), and every
    PR bin has its relative xy-coordinates with respect to the ground radar
    site.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_xyz : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins, 3). Contains corrected
        PR xy coordinates in GR azimuthal equidistant projection and altitude
    gr_site_alt : float
        Altitude of the GR site (in meters)
    re : float
        Effective Earth radius at GR site (in meters)

    Returns
    -------
    r : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the slant
        distance of PR bins from GR site.
    theta: :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the elevation
        angle of PR bins as seen from GR site.
    phi : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the azimuth
        angles of PR bins as seen from GR site.
    """
    # calculate arc length
    s = np.sqrt(np.sum(pr_xyz[..., 0:2] ** 2, axis=-1))

    # calculate arc angle
    gamma = s / re

    # calculate theta (elevation-angle)
    numer = np.cos(gamma) - (re + gr_site_alt) / (re + pr_xyz[..., 2])
    denom = np.sin(gamma)
    theta = np.rad2deg(np.arctan(numer / denom))

    # calculate SlantRange r
    r = (re + pr_xyz[..., 2]) * denom / np.cos(np.deg2rad(theta))

    # calculate Azimuth phi
    phi = 90 - np.rad2deg(np.arctan2(pr_xyz[..., 1], pr_xyz[..., 0]))
    phi[phi <= 0] += 360

    return r, theta, phi


def dist_from_orbit(pr_alt, alpha, r_pr_inv):
    """Returns range distances of PR bins (in meters) as seen from the orbit

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_alt : float
        PR orbit height in meters.
    alpha: :class:`numpy:numpy.ndarray`
       Array of depression angles of the PR beams with shape (nbeams).
    r_pr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to PR platform of shape (nbins).

    Returns
    -------
    ranges : :class:`numpy:numpy.ndarray`
        Array of shape (nbeams, nbins) of PR bin range distances from
        PR platform in orbit.
    """
    return pr_alt / np.cos(np.radians(alpha))[:, np.newaxis] - r_pr_inv


def create_raster_dataset(data, coords, projection=None, nodata=-9999):
    """ Create In-Memory Raster Dataset

    .. versionadded 0.10.0

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    projection : osr object
        Spatial reference system of the used coordinates, defaults to None.

    Returns
    -------
    dataset : gdal.Dataset
        In-Memory raster dataset

    Note
    ----
    The origin of the provided data and coordinates is UPPER LEFT.
    """

    # align data
    data = data.copy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, rows, cols = data.shape

    # create In-Memory Raster with correct dtype
    mem_drv = gdal.GetDriverByName('MEM')
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
    dataset = mem_drv.Create('', cols, rows, bands, gdal_type)

    # initialize geotransform
    x_ps, y_ps = coords[1, 1] - coords[0, 0]
    geotran = [coords[0, 0, 0], x_ps, 0, coords[0, 0, 1], 0, y_ps]
    dataset.SetGeoTransform(geotran)

    if projection:
        dataset.SetProjection(projection.ExportToWkt())

    # set np.nan to nodata
    dataset.GetRasterBand(1).SetNoDataValue(nodata)

    for i, band in enumerate(data, start=1):
        dataset.GetRasterBand(i).WriteArray(band)

    return dataset


def set_raster_origin(data, coords, direction):
    """ Converts Data and Coordinates Origin

    .. versionadded 0.10.0

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    direction : str
        'lower' or 'upper', direction in which to convert data and coordinates.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    """
    x_sp, y_sp = coords[1, 1] - coords[0, 0]
    origin = ('lower' if y_sp > 0 else 'upper')
    same = (origin == direction)
    if not same:
        data = np.flip(data, axis=-2)
        coords = np.flip(coords, axis=-3)
        # we need to shift y-coordinate if data and coordinates have the same
        # number of rows and cols
        if data.shape[-2:] == coords.shape[:2]:
            coords += [0, y_sp]

    return data, coords


def extract_raster_dataset(dataset, nodata=None):
    """ Extract data, coordinates and projection information

    Parameters
    ----------
    dataset : gdal.Dataset
        raster dataset
    nodata : scalar
        Value to which the dataset nodata values are mapped.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    projection : osr object
        Spatial reference system of the used coordinates.
    """

    # data values
    data = read_gdal_values(dataset, nodata=nodata)

    # coords
    coords_pixel = pixel_coordinates(dataset.RasterXSize,
                                     dataset.RasterYSize,
                                     'edges')
    coords = pixel_to_map(dataset.GetGeoTransform(),
                          coords_pixel)

    projection = read_gdal_projection(dataset)

    return data, coords, projection


def _doctest_():
    import doctest
    print('doctesting')
    doctest.testmod()
    print('finished')


if __name__ == '__main__':
    print('wradlib: Calling module <georef> as main...')
