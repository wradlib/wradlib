# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        georef
# Purpose:
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     26.10.2011
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

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
   create_projstr
   proj4_to_osr
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

"""

##* Seitenlänge Zenit - Himmelsnordpol: 90°-phi
##* Seitenlänge Himmelsnordpol - Gestirn: 90°-delta
##* Seitenlänge Zenit - Gestirn: 90°-h
##* Winkel Himmelsnordpol - Zenit - Gestirn: 180°-a
##* Winkel Zenit - Himmelsnordpol - Gestirn: tau

## alpha - rektaszension
## delta - deklination
## theta - sternzeit
## tau = theta - alpha - stundenwinkel
## a - azimuth (von süden aus gezählt)
## h - Höhe über Horizont

from osgeo import gdal,osr
#from numpy import sin, cos, arcsin, pi
import numpy as np
from sys import exit
import warnings


def hor2aeq(a, h, phi):
    """"""
    delta = np.arcsin(np.sin(h)*np.sin(phi) - np.cos(h)*np.cos(a)*np.cos(phi))
    tau = np.arcsin(np.cos(h)*np.sin(a)/np.cos(delta))
    return delta, tau

def aeq2hor(tau, delta, phi):
    """"""
    h = np.arcsin(np.cos(delta)*np.cos(tau)*np.cos(phi) + np.sin(delta)*np.sin(phi))
    a = np.arcsin(np.cos(delta)*np.sin(tau)/np.cos(h))
    return a, h


def polar2lonlat(r, az, sitecoords, re=6370040):
    """Transforms polar coordinates (of a PPI) to longitude/latitude \
    coordinates.

    This function assumes that the transformation from the polar radar
    coordinate system to the earth's spherical coordinate system may be done
    in the same way as astronomical observations are transformed from the
    horizon's coordinate system to the equatorial coordinate system.

    The conversion formulas used were taken from
    http://de.wikipedia.org/wiki/Nautisches_Dreieck [accessed 2001-11-02] and
    are only valid as long as the radar's elevation angle is small, as one main
    assumption of this method is, that the 'zenith-star'-side of the nautic
    triangle can be described by the radar range divided by the earths radius.
    For lager elevation angles, this side     would have to be reduced.

    Parameters
    ----------
    r : array
        array of ranges [m]
    az : array
        array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location
    re : float
        earth's radius [m]

    Returns
    -------
    lon, lat : tuple of arrays
        two arrays containing the spherical longitude and latitude coordinates

    Notes
    -----
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
    ...     print '{0:6.2f}, {1:6.2f}'.format(x, y)
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

    #phi = 48.58611111 * pi/180.  # drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #lon = 9.783888889 * pi/180.  # drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    phi = np.deg2rad(sitecoords[1])
    lam = np.deg2rad(sitecoords[0])

    a = np.deg2rad(-(180. + az))
    h =  0.5*np.pi - r/re

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
    ...     print '{0:6.2f}, {1:6.2f}'.format(x, y)
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

    l = np.deg2rad(90.-phia)
    r = rng/re

    easterly = az<=180.
    westerly = ~easterly
    a = np.deg2rad(np.where(easterly,az,az-180.))

    m = np.arccos(np.cos(r)*np.cos(l) + np.sin(r)*np.sin(l)*np.cos(a))
    g = np.arcsin((np.sin(r)*np.sin(a))/(np.sin(m)))

    return thea+np.rad2deg(np.where(easterly,g,-g)), 90.-np.rad2deg(m)


def polar2lonlatalt(r, az, elev, sitecoords, re=6370040.):
    """Transforms polar coordinates to lon/lat/altitude coordinates.

    Explicitely accounts for the beam's elevation angle and for the altitude of the radar location.

    This is an alternative implementation based on VisAD code (see
    http://www.ssec.wisc.edu/visad-docs/javadoc/visad/bom/Radar3DCoordinateSystem.html#toReference%28float[][]%29 and
    http://www.ssec.wisc.edu/~billh/visad.html ).

    VisAD code has been translated to Python from Java.

    Nomenclature tries to stick to VisAD code for the sake of comparibility, however, names of
    arguments are the same as for polar2lonlat...

    Parameters
    ----------
    r : array
        array of ranges [m]
    az : array
        array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of three floats
        the lon / lat coordinates of the radar location and its altitude a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    re : float
        earth's radius [m]

    Returns
    -------
    output : a tuple of three arrays (longitudes, latitudes,  altitudes)

    Examples
    --------
    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])*1000
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> th = np.array([0.,   0.,   0.,   0.,   0.,  0.5,])
    >>> csite = (9.0, 48.0)
    >>> lon1, lat1, alt1 = polar2lonlatalt(r, az, th, csite)
    >>> for x, y, z in zip(lon1, lat1, alt1):
    ...     print '{0:7.4f}, {1:7.4f}, {2:7.4f}'.format(x, y, z)
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

    cosaz = np.cos( np.deg2rad(az) )
    sinaz = np.sin( np.deg2rad(az) )
    # assume azimuth = 0 at north, then clockwise
    coselev = np.cos( np.deg2rad(elev) )
    sinelev = np.sin( np.deg2rad(elev) )
    rp = np.sqrt(re * re + r * r + 2.0 * sinelev * re * r)

    # altitudes
    alts = rp - re + centalt

    angle = np.arcsin(coselev * r / rp) # really sin(elev+90)
    radp = re * angle
    lats = centlat + cosaz * radp / _latscale()
    lons = centlon + sinaz * radp / _lonscale(centlat)

    return lons, lats, alts


def _latscale(re=6370040.):
    """Return meters per degree latitude assuming spherical earth
    """
    return 2*np.pi*re / 360.


def _lonscale(lat, re=6370040.):
    """Return meters per degree longitude assuming spherical earth
    """
    return (2*np.pi*re / 360.) * np.cos( np.deg2rad(lat) )


def beam_height_n(r, theta, re=6370040., ke=4./3.):
    r"""Calculates the height of a radar beam taking the refractivity of the
    atmosphere into account.

    Based on Doviak :cite:`Doviak1993` the beam height is calculated as

    .. math::

        h = \sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \sin\theta} - k_e r_e


    Parameters
    ----------
    r : array
        array of ranges [m]
    theta : scalar or array broadcastable to the shape of r
        elevation angles in degrees with 0° at horizontal and +90°
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
    return np.sqrt( r**2 + (ke*re)**2 + 2*r*ke*re*np.sin(np.radians(theta)) ) - ke*re


def arc_distance_n(r, theta, re=6370040., ke=4./3.):
    r"""Calculates the great circle distance of a radar beam over a sphere,
    taking the refractivity of the atmosphere into account.

    Based on Doviak :cite:`Doviak1993` the arc distance may be calculated as

    .. math::

        s = k_e r_e \arcsin\left(\frac{r \cos\theta}{k_e r_e + h_n(r, \theta, r_e, k_e)}\right)

    where :math:`h_n` would be provided by beam_height_n

    Parameters
    ----------
    r : array
        array of ranges [m]
    theta : scalar or array broadcastable to the shape of r
        elevation angles in degrees with 0° at horizontal and +90°
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
    return ke*re * np.arcsin((r*np.cos(np.radians(theta)))/
                             (ke*re + beam_height_n(r, theta, re, ke)))


def polar2lonlatalt_n(r, az, elev, sitecoords, re=None, ke=4./3.):
    """Transforms polar coordinates (of a PPI) to longitude/latitude \
    coordinates taking elevation angle and refractivity into account.

    It takes the shortening of the great circle
    distance with increasing  elevation angle as well as the resulting
    increase in height into account.

    Parameters
    ----------
    r : array
        array of ranges [m]
    az : array
        array of azimuth angles containing values between 0 and 360°.
        These are assumed to start with 0° pointing north and counted
        positive clockwise!
    th : scalar or array of the same shape as az
        elevation angles in degrees starting with 0° at horizontal to 90°
        pointing vertically upwards from the radar
    sitecoords : a sequence of two or three floats
        the lon / lat coordinates of the radar location, the third value,
        if present, will be interpreted as the height of the site above the
        geoid (i.e. sphere)
    re : float
        earth's radius [m], if None, `get_earth_radius` will be used to
        determine the equivalent radius of the WGS84 ellipsoid for the
        latitude given in sitecoords.
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    lon, lat, alt : tuple of arrays
        three arrays containing the spherical longitude and latitude coordinates
        as well as the altitude of the beam.

    Notes
    -----
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
    ...     print '{0:7.4f}, {1:7.4f}, {2:7.4f}'.format(x, y, z)
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
    x = arc * np.cos(np.radians(90-az))
    y = arc * np.sin(np.radians(90-az))

    # then it's just a matter of invoking reproject
    lon, lat = reproject(x, y, projection_source=rad, projection_target=sph)

    return lon, lat, alt


def centroid2polyvert(centroid, delta):
    """Calculates the 2-D Polygon vertices necessary to form a rectangular polygon around the centroid's coordinates.

    The vertices order will be clockwise, as this is the convention used
    by ESRI's shapefile format for a polygon.

    Parameters
    ----------
    centroid : array_like
               list of 2-D coordinates of the center point of the rectangle
    delta :    scalar or array
               symmetric distances of the vertices from the centroid in each
               direction. If `delta` is scalar, it is assumed to apply to
               both dimensions.

    Returns
    -------
    vertices : array
               an array with 5 vertices per centroid.

    Notes
    -----
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
    if (cent.shape[0]==2) and (cent.shape[-1]!=2):
        cent = np.transpose(cent)
    assert cent.shape[-1] == 2
    dshape = [1]*cent.ndim
    dshape.insert(-1, 5)
    dshape[-1] = 2

    d = np.array([[-1.,-1.],
                  [-1.,1.],
                  [1., 1.],
                  [1.,-1.],
                  [-1.,-1.]]).reshape(tuple(dshape))

    return np.asanyarray(centroid)[...,None,:] + d * np.asanyarray(delta)


def polar2polyvert(r, az, sitecoords):
    """
    Generate 2-D polygon vertices directly from polar coordinates.

    This is an alternative to centroid2polyvert which does not use centroids,
    but generates the polygon vertices by simply connecting the corners of the
    radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    polar2lonlat.

    Parameters
    ----------

    r : array
        array of ranges [m]; r defines the exterior boundaries of the range bins!
        (not the centroids). Thus, values must be positive!
    az : array
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main beam lobe!
        The first angle can start at any values, but make sure the array is sorted
        continuously positively clockwise and the angles are equidistant. An angle
        if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location

    Returns
    -------
    output : a 3-d array of polygon vertices in lon/lat
        with shape(num_vertices, num_vertex_nodes, 2). The last dimension
        carries the longitudes on the first position, the latitudes on the
        second (lon: output[:,:,0], lat: output[:,:,1]

    Examples
    --------
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
    >>> polygons = polar2polyvert(r, az, sitecoords)
    >>> # plot the resulting mesh
    >>> fig = pl.figure()
    >>> ax = fig.add_subplot(111)
    >>> #polycoll = mpl.collections.PolyCollection(vertices,closed=True, facecolors=None)
    >>> polycoll = mpl.collections.PolyCollection(polygons,closed=True, facecolors=None)
    >>> ret = ax.add_collection(polycoll, autolim=True)
    >>> #pl.axis('tight')
    >>> pl.show()

    """
    # prepare the range and azimuth array so they describe the boundaries of a bin,
    #   not the centroid
    r, az = _check_polar_coords(r,az)
    r = np.insert(r, 0, r[0] - _get_range_resolution(r) )
    az = az - 0.5*_get_azimuth_resolution(az)
    az = np.append(az, az[0])
    az = np.where(az<0, az+360., az)

    # generate a grid of polar coordinates of bin corners
    r, az = np.meshgrid(r, az)
    # convert polar coordinates to lat/lon
    lon, lat= polar2lonlat(r, az, sitecoords)

    llc = np.transpose(np.vstack((lon[:-1,:-1].ravel(), lat[:-1,:-1].ravel())))
    ulc = np.transpose(np.vstack((lon[:-1,1: ].ravel(), lat[:-1,1: ].ravel())))
    urc = np.transpose(np.vstack((lon[1: ,1: ].ravel(), lat[1: ,1: ].ravel())))
    lrc = np.transpose(np.vstack((lon[1: ,:-1].ravel(), lat[1: ,:-1].ravel())))

    vertices = np.concatenate((llc, ulc, urc, lrc, llc)).reshape((-1,5,2), order='F')

    return vertices




def polar2centroids(r=None, az=None, sitecoords=None, range_res = None):
    """
    Computes the lat/lon centroids of the radar bins from the polar coordinates.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. The ranges are assumed to define the exterior boundaries
    of the range bins (thus they must be positive). The angles are assumed to
    describe the pointing direction fo the main beam lobe.

    For further information refer to the documentation of georef.polar2lonlat.

    r : array
        array of ranges [m]; r defines the exterior boundaries of the range bins!
        (not the centroids). Thus, values must be positive!
    az : array
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main beam lobe!
        The first angle can start at any values, but make sure the array is sorted
        continuously positively clockwise and the angles are equidistant. An angle
        if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lon / lat coordinates of the radar location
    range_res : float
        range resolution of radar measurement [m] in case it cannot be derived
        from r (single entry in r-array)

    Returns
    -------
    output : tuple of 2 arrays which describe the bin centroids
        longitude and latitude

    Notes
    -----
    Azimuth angles of 360 deg are internally converted to 0 deg.

    """
    # make sure the range and azimuth angles have the right properties
    r, az = _check_polar_coords(r, az)

    # to get the centroid, we only have to move the exterior bin boundaries by half the resolution
    if range_res:
        r = r - 0.5 * range_res
    else:
        r = r - 0.5*_get_range_resolution(r)
    # generate a polar grid and convert to lat/lon
    r, az = np.meshgrid(r, az)
    lon, lat= polar2lonlat(r, az, sitecoords)

    return lon, lat


def _check_polar_coords(r, az):
    """
    Contains a lot of checks to make sure the polar coordinates are adequate.

    Parameters
    ----------
    r : range gates in any unit
    az : azimuth angles in degree

    """
    r = np.array(r, 'f4')
    az = np.array(az, 'f4')
    az[az==360.] = 0.
    if 0. in r:
        print('Invalid polar coordinates: 0 is not a valid range gate specification (the centroid of a range gate must be positive).')
        exit()
    if len(np.unique(r))!=len(r):
        print('Invalid polar coordinates: Range gate specification contains duplicate entries.')
        exit()
    if len(np.unique(az))!=len(az):
        print('Invalid polar coordinates: Azimuth specification contains duplicate entries.')
        exit()
    if len(np.unique(az))!=len(az):
        print('Invalid polar coordinates: Azimuth specification contains duplicate entries.')
        exit()
    if not _is_sorted(r):
        print('Invalid polar coordinates: Range array must be sorted.')
        exit()
    if len(np.unique(r[1:]-r[:-1]))>1:
        print('Invalid polar coordinates: Range gates are not equidistant.')
        exit()
    if len(np.where(az>=360.)[0])>0:
        print('Invalid polar coordinates: Azimuth angles must not be greater than or equal to 360 deg.')
        exit()
    if not _is_sorted(az):
        # it is ok if the azimuth angle array is not sorted, but it has to be
        #   'continuously clockwise', e.g. it could start at 90° and stop at °89
        az_right = az[np.where(np.logical_and(az<=360, az>=az[0]))[0]]
        az_left = az[np.where(az<az[0])]
        if ( not _is_sorted(az_right) ) or ( not _is_sorted(az_left) ):
            print('Invalid polar coordinates: Azimuth array is not sorted clockwise.')
            exit()
    if len(np.unique(np.sort(az)[1:] - np.sort(az)[:-1]))>1:
        warnings.warn("The azimuth angles of the current dataset are not equidistant.", UserWarning)
##        print 'Invalid polar coordinates: Azimuth angles are not equidistant.'
##        exit()
    return r, az


def _is_sorted(x):
    """
    Returns True when array x is sorted
    """
    return np.all(x==np.sort(x))


def _get_range_resolution(x):
    """
    Returns the range resolution based on
    the array x of the range gates' exterior limits
    """
    if len(x)<=1:
        print 'The range gate array has to contain at least two values for deriving the resolution.'
        exit()
    res = np.unique(x[1:]-x[:-1])
    if len(res)>1:
        print 'The resolution of the range array is ambiguous.'
        exit()
    return res[0]

def _get_azimuth_resolution(x):
    """
    Returns the azimuth resolution based on the array x of the beams' azimuth angles
    """
    res = np.unique(np.sort(x)[1:]-np.sort(x)[:-1])
    if len(res)>1:
        print 'The resolution of the azimuth angle array is ambiguous.'
        exit()
    return res[0]


def create_projstr(projname, **kwargs):
    """Conveniently supports the construction of proj.4 projection strings

    Currently, the following projection names (argument *projname*) are supported:

    **"aeqd": Azimuthal Equidistant**

    needs the following keyword arguments: *lat_0* (latitude at projection center),
    *lon_0* (longitude at projection center), *x_0* (false Easting, also known as x-offset),
    *y_0* (false Northing, also known as y-offset)

    **"gk" : Gauss-Krueger (for Germany)**

    only needs keyword argument *zone* (number of the Gauss-Krueger strip)

    **"utm" : Universal Transmercator**

    needs keyword arguments *zone* (integer) and optionally *hemisphere* (accepted values: "south", "north")
    see `Wikipedia entry <http://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>`_ for UTM zones.

    **"dwd-radolan" : RADOLAN Composite Coordinate System**

    no additional arguments needed.

    Polar stereographic projection used by the German Weather Service (DWD)
    for all Radar composite products. See the final report on the RADOLAN
    project (available at http://www.dwd.de/RADOLAN) for details.

    Parameters
    ----------
    projname : string (proj.4 projection acronym)
    kwargs : depends on projname - see above!

    Returns
    -------
    output : string (a proj.4 projection string)

    Examples
    --------
    >>> # Gauss-Krueger 2nd strip
    >>> print create_projstr("gk", zone=2)
    +proj=tmerc +lat_0=0 +lon_0=6 +k=1 +x_0=2500000 +y_0=0
                +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7
                +units=m +no_defs
    >>> # UTM zone 51 (northern hemisphere)
    >>> print create_projstr("utm", zone=51)
    +proj=utm +zone=51 +ellps=WGS84
    >>> # UTM zone 51 (southern hemisphere)
    >>> print create_projstr("utm", zone=51, hemisphere="south")
    +proj=utm +zone=51 +ellps=WGS84 +south

    """
    if projname=="aeqd":
        # Azimuthal Equidistant
        if "x_0" in kwargs:
            projstr = """+proj=aeqd  +lat_0=%f +lon_0=%f +x_0=%f +y_0=%f""" \
                  % (kwargs["lat_0"], kwargs["lon_0"], kwargs["x_0"], kwargs["y_0"])
        else:
            projstr = """+proj=aeqd  +lat_0=%f +lon_0=%f""" \
                  % (kwargs["lat_0"], kwargs["lon_0"])
    elif projname=="gk":
        # Gauss-Krueger
        if kwargs.has_key("zone"):
            projstr = """+proj=tmerc +lat_0=0 +lon_0=%d +k=1 +x_0=%d +y_0=0
            +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7
            +units=m +no_defs""" % (kwargs["zone"]*3,
                                    kwargs["zone"] * 1000000 + 500000)
    elif projname=="utm":
        # Universal Transmercator
        if kwargs.has_key("hemisphere"):
            if kwargs["hemisphere"]=="south":
                hemisphere = " +south"
            elif kwargs["hemisphere"]=="north":
                hemisphere = ""
            else:
                print "Value %s for keyword argument hemisphere in function create_projstr is not valid. Value must be either north or south!" % kwargs["hemisphere"]
                exit(1)
        else:
            hemisphere = ""
        try:
            projstr = "+proj=utm +zone=%d +ellps=WGS84%s" % (kwargs["zone"], hemisphere)
        except:
            print "Cannot create projection string for projname %s. Maybe keyword argument zone was not passed?" % projname
            exit(1)

    elif projname=="dwd-radolan":
        # DWD-RADOLAN polar stereographic projection
        scale = (1.+np.sin(np.radians(60.)))/(1.+np.sin(np.radians(90.)))
        projstr = ('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k={0:10.8f} '
                   '+x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs').format(scale)
    else:
        print "No support for projection %r, yet." % projname
        print "You need to create projection string by hand..."
        exit(1)
    return projstr


def projected_bincoords_from_radarspecs(r, az, sitecoords, projstr, range_res = None):
    """
    Convenience function to compute projected bin coordinates directly from
    radar site coordinates and range/azimuth specs

    Parameters
    ----------
    r : array
        array of ranges [m]; r defines the exterior boundaries of the range bins!
        (not the centroids). Thus, values must be positive!
    az : array
    sitecoords : tuple
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main beam lobe!
        The first angle can start at any values, but make sure the array is sorted
        continuously positively clockwise and the angles are equidistant. An angle
        if 0 degree is pointing north.
    projstr : string
        proj.4 projection string
    range_res : float
        range resolution of radar measurement [m] in case it cannot be derived
        from r (single entry in r-array)

    """
    cent_lon, cent_lat = polar2centroids(r, az, sitecoords, range_res = range_res)
    osr_proj = proj4_to_osr(projstr)
    x, y = reproject(cent_lon, cent_lat, projection_target = osr_proj)
    return x.ravel(), y.ravel()


def get_earth_radius(latitude, sr= None):
    """
    Get the radius of the Earth (in km) for a given Spheroid model (sr) at a given position

    R^2 = ( a^4 cos(f)^2 + b^4 sin(f)^2 ) / ( a^2 cos(f)^2 + b^2 sin(f)^2 ).

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
    radius = np.sqrt((np.power(RADIUS_E,4) * np.power(np.cos(latitude),2) + np.power(RADIUS_P,4) * np.power(np.sin(latitude),2) ) / ( np.power(RADIUS_E,2) * np.power(np.cos(latitude),2) +  np.power(RADIUS_P,2) * np.power(np.sin(latitude),2) ))
    return(radius)


def pixel_coordinates(nx,ny,mode="centers"):
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
    coordinates : np array
         array of shape (ny,nx) with pixel coordinates (x,y)

    """
    if mode == "centroids":
        mode = "centers"
    x = np.linspace(0,nx,num=nx+1)
    y = np.linspace(0,ny,num=ny+1)
    if  mode == "centers":
        x = x + 0.5
        y = y + 0.5
        x = np.delete(x,-1)
        y = np.delete(y,-1)
    X,Y = np.meshgrid(x,y)
    coordinates = np.empty(X.shape + (2,))
    coordinates[:,:,0] = X
    coordinates[:,:,1] = Y
    return (coordinates)


def pixel_to_map(geotransform,coordinates):
    """Apply a geographical transformation to return map coordinates from pixel coordinates.

    Parameters
    ----------
    geotransform : np array
        geographical transformation vector:
            geotransform[0] = East/West location of Upper Left corner
            geotransform[1] = X pixel size
            geotransform[2] = X pixel rotation
            geotransform[3] = North/South location of Upper Left corner
            geotransform[4] = Y pixel rotation
            geotransform[5] = Y pixel size
    coordinates : 2d array
        array of pixel coordinates

    Returns
    -------
    coordinates_map : np array
        3d array with map coordinates x,y
    """
    coordinates_map = np.empty(coordinates.shape)
    coordinates_map[...,0] = geotransform[0] + geotransform[1] * coordinates[...,0] + geotransform[2] * coordinates[...,1]
    coordinates_map[...,1] = geotransform[3] + geotransform[4] * coordinates[...,0] + geotransform[5] * coordinates[...,1]
    return(coordinates_map)


def pixel_to_map3d(geotransform, coordinates, z=None):
    """Apply a geographical transformation to return 3D map coordinates from pixel coordinates.

    Parameters
    ----------
    geotransform : np array
        geographical transformation vector (see pixel_to_map())
    coordinates : 2d array
        array of pixel coordinates;
    z : string
        method to compute the z coordinates (height above ellipsoid) :
            None : default, z equals zero
            srtm : not available yet

    Returns
    -------
    coordinates_map : 4d array
        4d array with map coordinates x,y,z

    """

    coordinates_map = np.empty(coordinates.shape[:-1] + (3,))
    coordinates_map[...,0:2] = pixel_to_map(geotransform, coordinates)
    coordinates_map[...,2] = np.zeros(coordinates.shape[:-1])
    return(coordinates_map)


def read_gdal_coordinates(dataset,mode='centers',z=True):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal object
        raster image with georeferencing
    mode : string
        either 'centers' or 'borders'
    z : boolean
        True to get height coordinates (zero).

    Returns
    -------
    coordinates : 3D np array
        projected coordinates (x,y,z)
    """
    coordinates_pixel = pixel_coordinates(dataset.RasterXSize,dataset.RasterYSize,mode)
    geotransform = dataset.GetGeoTransform()
    if z:
        coordinates = pixel_to_map3d(geotransform,coordinates_pixel)
    else:
        coordinates = pixel_to_map(geotransform,coordinates_pixel)
    return(coordinates)


def read_gdal_projection(dset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dset : gdal dataset object

    Returns
    -------
    srs : OSR object
        dataset projection
    """
    proj4 = dset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4)
    src = None
    return(srs)


def read_gdal_values(data=None,nodata=False):
    """Read values from a gdal object.

    Parameters
    ----------
    data : gdal object
    nodata : boolean
        option to deal with nodata values replacing it with nans.

    Returns
    -------
    values : 2d array
        array with values
    """

    b1 = data.GetRasterBand(1)
    values = b1.ReadAsArray()
    if nodata:
        nodata = b1.GetNoDataValue()
        values = values.astype('float')
        values[values==nodata] = np.nan
    return(values)

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

    C : multidimensional np array
        array of shape (...,2) or (...,3) with coordinates (x,y) or (x,y,z)
        respectively
    X : nd array
        array of x coordinates
    Y : nd array
        array of y coordinates
    Z : nd array
        array of z coordinates

    Keyword arguments:
    projection_source : osr object (defaults to EPSG(4326)
    projection_target : osr object (defaults to EPSG(4326)

    Returns
    -------
    trans : nd array
        array of reprojected coordinates x,y (...,2) or x,y,z (...,3)
        depending on input array
    X, Y : nd arrays
        arrays of reprojected x,y coordinates, shape depending on input array
    X, Y, Z: nd arrays
        arrays of reprojected x,y,z coordinates, shape depending on input array
    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape =  C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1,numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError('Input Array column mismatch to %s' % ('reproject'))
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
            C = np.concatenate([X.ravel()[:,None],
                                Y.ravel()[:,None],
                                Z.ravel()[:,None]], axis=1)
        else:
            C = np.concatenate([X.ravel()[:,None],
                                Y.ravel()[:,None]], axis=1)

    projection_source = kwargs.get('projection_source', get_default_projection())
    projection_target = kwargs.get('projection_target', get_default_projection())

    ct = osr.CoordinateTransformation(projection_source,projection_target)
    trans = np.array(ct.TransformPoints(C))

    if len(args) == 1:
        # here we could do this one
        #return(np.array(ct.TransformPoints(C))[...,0:numCols]))
        # or this one
        trans = trans[:,0:numCols].reshape(cshape)
        return trans
    else:
        X = trans[:,0].reshape(xshape)
        Y = trans[:,1].reshape(yshape)
        if len(args) == 2:
            return X, Y
        if len(args) == 3:
            Z = trans[:,2].reshape(zshape)
            return X, Y, Z

def get_default_projection():
    """Create a default projection object (wgs84)"""
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    return(proj)


def sweep_centroids(nrays,rscale,nbins,elangle):
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
        array of shape (nrays,nbins,3) containing native centroid radar coordinates (slant range, azimuth, elevation)
    """
    ascale = np.pi/nrays
    azimuths = ascale/2 + np.linspace(0,2*np.pi,nrays,endpoint=False)
    ranges = np.arange(nbins)*rscale + rscale/2
    coordinates = np.empty((nrays,nbins,3),dtype=float)
    coordinates[:,:,0] = np.tile(ranges,(nrays,1))
    coordinates[:,:,1] = np.transpose(np.tile(azimuths,(nbins,1)))
    coordinates[:,:,2] = elangle
    return(coordinates)


def proj4_to_osr(proj4str):
    """Transform a proj4 string to an osr spatial reference object"""
    if proj4str:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(proj4str)
    else:
        proj = get_default_projection()
    return(proj)

def get_radolan_coords(lon, lat, trig=False):
    """
    Calculates x,y coordinates of radolan grid from lon, lat

    Parameters
    ----------

    lon :   float, array of floats
        longitude
    lat :   float, array of floats
        latitude
    trig : boolean
        if True, uses trigonometric formulas for calculation, otherwise osr transformations
        if False, uses osr spatial reference system to transform between projections
        `trig` is recommended to be False, however, the two ways of computation are expected
        to be equivalent.
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
        dwd_string = create_projstr("dwd-radolan")
        proj_stereo = proj4_to_osr(dwd_string)

        # create wgs84 projection osr object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        x, y = reproject(lon, lat, projection_source=proj_wgs, projection_target=proj_stereo)

    return x, y


def get_radolan_grid(nrows=None, ncols=None, trig=False, wgs84=False):
    """Calculates x/y coordinates of radolan grid of the German Weather Service

    Returns the x,y coordinates of the radolan grid positions
    (lower left corner of every pixel). The radolan grid is a polarstereographic
    projection, the projection information was taken from RADOLAN-RADVOR-OP
    Kompositformat_2.2.2  :cite:`DWD2009`

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

    .. table:: Coordinates for 1500km x 1400km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  2.3419E  |  43.9336N  | -673.4622 | -5008.645 |
        +------------+-----------+------------+-----------+-----------+

    Parameters
    ----------
    nrows : int
        number of rows (460, 900 by default, 1500)
    ncols : int
        number of columns (460, 900 by default, 1400)
    trig : boolean
        if True, uses trigonometric formulas for calculation
        if False, uses osr spatial reference system to transform between projections
        `trig` is recommended to be False, however, the two ways of computation are expected
        to be equivalent.
    wgs84 : boolean
        if True, output coordinates are in wgs84 lonlat format (default: False)

    Returns
    -------
    radolan_grid : numpy ndarray (rows, cols, 2)
                   xy- or lonlat-grid

    Examples
    --------

        >>> # using osr spatial reference transformation
        >>> import wradlib.georef as georef
        >>> radolan_grid = georef.get_radolan_grid()
        >>> print(radolan_grid.shape, radolan_grid[0,0,:])
        ((900, 900, 2), array([ -523.46216677, -4658.64471573]))

        >>> # using pure trigonometric transformations
        >>> import wradlib.georef as georef
        >>> radolan_grid = georef.get_radolan_grid(trig=True)
        >>> print(radolan_grid.shape, radolan_grid[0,0,:])
        ((900, 900, 2), array([ -523.46216692, -4658.64472427]))

        >>> # using osr spatial reference transformation
        >>> import wradlib.georef as georef
        >>> radolan_grid = georef.get_radolan_grid(1500, 1400)
        >>> print(radolan_grid.shape, radolan_grid[0,0,:])
        ((1500, 1400, 2), array([ -673.46216677, -5008.64471573]))

        >>> # using osr spatial reference transformation
        >>> import wradlib.georef as georef
        >>> radolan_grid = georef.get_radolan_grid(900, 900, wgs84=True)
        >>> print(radolan_grid.shape, radolan_grid[0,0,:])
        ((900, 900, 2), array([  3.58892994,  46.9525804 ]))

    Raises
    ------
        TypeError, ValueError

    """

    # setup default parameters in dicts
    small = {'j_0': 460, 'i_0': 460, 'res': 2}
    normal = {'j_0': 450, 'i_0': 450, 'res': 1}
    extended = {'j_0': 600, 'i_0': 800, 'res': 1}
    griddefs = {(460, 460): small, (900, 900): normal, (1500, 1400): extended}

    # type and value checking
    if nrows and ncols:
        if not (isinstance(nrows, int) and isinstance(ncols, int)):
            raise TypeError("wradlib.georef: Parameter *nrows* and *ncols* not integer")
        if (nrows, ncols) not in griddefs.iterkeys():
            raise ValueError("wradlib.georef: Parameter *nrows* and *ncols* mismatch.")
    else:
        # fallback for call without parameters
        nrows = 900
        ncols = 900

    # small, normal or extended grid check
    # reference point changes according to radolan composit format
    j_0 = griddefs[(nrows, ncols)]['j_0']
    i_0 = griddefs[(nrows, ncols)]['i_0']
    res = griddefs[(nrows, ncols)]['res']

    x_0, y_0 = get_radolan_coords(9.0, 51.0, trig=trig)

    x_arr = np.arange(x_0 - j_0, x_0 - j_0 + ncols, res)
    y_arr = np.arange(y_0 - i_0, y_0 - i_0 + nrows, res)
    x, y = np.meshgrid(x_arr, y_arr)

    radolan_grid = np.dstack((x, y))

    if wgs84:

        if trig:
            # inverse projection
            lon0 = 10.   # central meridian of projection
            lat0 = 60.   # standard parallel of projection

            sinlat0 = np.sin(np.radians(lat0))

            fac = (6370.040**2.) * ((1.+sinlat0) ** 2.)
            lon = np.degrees(np.arctan((-x/y))) + lon0
            lat = np.degrees(np.arcsin((fac - (x**2. + y**2.))/(fac + (x**2. + y**2.))))
            radolan_grid = np.dstack((lon, lat))
        else:
            # create radolan projection osr object
            dwd_string = create_projstr("dwd-radolan")
            proj_stereo = proj4_to_osr(dwd_string)

            # create wgs84 projection osr object
            proj_wgs = osr.SpatialReference()
            proj_wgs.ImportFromEPSG(4326)

            radolan_grid = reproject(radolan_grid, projection_source=proj_stereo, projection_target=proj_wgs)

    return radolan_grid


def _doctest_():
    import doctest
    print 'doctesting'
    doctest.testmod()
    print 'finished'


if __name__ == '__main__':
    print 'wradlib: Calling module <georef> as main...'
