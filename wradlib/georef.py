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

   polar2latlon
   polar2latlonalt
   beam_height_n
   arc_distance_n
   polar2latlonalt_n
   polar2centroids
   polar2polyvert
   centroid2polyvert
   project
   create_projstr
   projected_bincoords_from_radarspecs

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

from numpy import sin, cos, arcsin, pi
import numpy as np
import pyproj
from sys import exit
import warnings


def hor2aeq(a, h, phi):
    """"""
    delta = arcsin(-cos(h)*cos(a)*cos(phi) + sin(h)*sin(phi))
    tau = arcsin(cos(h)*sin(a)/cos(delta))
    return delta, tau


def aeq2hor(tau, delta, phi):
    """"""
    h = arcsin(cos(delta)*cos(tau)*cos(phi) + sin(delta)*sin(phi))
    a = arcsin(cos(delta)*sin(tau)/cos(h))


def polar2latlon(r, az, sitecoords, re=6370040):
    """Transforms polar coordinates (of a PPI) to latitude/longitude \
    coordinates.

    This function assumes that the transformation from the polar radar
    coordinate system to the earth's spherical coordinate system may be done
    in the same way as astronomical observations are transformed from the
    horizon's coordinate system to the equatorial coordinate system.

    The conversion formulas used were taken from
    http://de.wikipedia.org/wiki/Nautisches_Dreieck [accessed 2001-11-02] and
    are
    only valid as long as the radar's elevation angle is small, as one main
    assumption of this method is, that the 'zenith-star'-side of the nautic
    triangle
    can be described by the radar range divided by the earths radius.
    For lager elevation angles, this side
    would have to be reduced.

    Parameters
    ----------
    r : array
        array of ranges [m]
    az : array
        array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of two floats
        the lat / lon coordinates of the radar location
    re : float
        earth's radius [m]

    Returns
    -------
    lat, lon : tuple of arrays
        two arrays containing the spherical latitude and longitude coordinates

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

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> csite = (48.0, 9.0)
    >>> lat1, lon1= __pol2latlon(r, az, csite)
    >>> for x, y in zip(lat1, lon1):
    ...     print '{0:6.2f}, {1:6.2f}'.format(x, y)
     48.00,   9.00
     48.00,   9.00
     49.00,   9.00
     47.99,  10.49
     47.00,   9.00
     47.99,   7.51

    The coordinates of the east and west directions won't come to lie on the
    latitude of the site because doesn't travel along the latitude circle but
    along a great circle.


    """

    #phi = 48.58611111 * pi/180.  # drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #lon = 9.783888889 * pi/180.  # drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    phi = np.deg2rad(sitecoords[0])
    lam = np.deg2rad(sitecoords[1])

    a   = np.deg2rad(-(180. + az))
    h   =  0.5*pi - r/re

    delta, tau = hor2aeq(a, h, phi)
    latc = np.rad2deg(delta)
    lonc = np.rad2deg(lam + tau)

    return latc, lonc


def __pol2latlon(rng, az, sitecoords, re=6370040):
    """Alternative implementation using spherical geometry only.

    apparently it produces the same results as polar2latlon.
    I wrote it because I suddenly doubted that the assumptions of the nautic
    triangle were wrong. I leave it here, in case someone might find it useful.

    Examples
    --------

    A few standard directions (North, South, North, East, South, West) with
    different distances (amounting to roughly 1°) from a site
    located at 48°N 9°E

    >>> r  = np.array([0.,   0., 111., 111., 111., 111.,])
    >>> az = np.array([0., 180.,   0.,  90., 180., 270.,])
    >>> csite = (48.0, 9.0)
    >>> lat1, lon1= __pol2latlon(r, az, csite)
    >>> for x, y in zip(lat1, lon1):
    ...     print '{0:6.2f}, {1:6.2f}'.format(x, y)
     48.00,   9.00
     48.00,   9.00
     49.00,   9.00
     47.99,  10.49
     47.00,   9.00
     47.99,   7.51

    The coordinates of the east and west directions won't come to lie on the
    latitude of the site because doesn't travel along the latitude circle but
    along a great circle.

    """
    phia = sitecoords[0]
    thea = sitecoords[1]

    l = np.deg2rad(90.-phia)
    r = rng/re

    easterly = az<=180.
    westerly = ~easterly
    a = np.deg2rad(np.where(easterly,az,az-180.))

    m = np.arccos(np.cos(r)*np.cos(l) + np.sin(r)*np.sin(l)*np.cos(a))
    g = np.arcsin((np.sin(r)*np.sin(a))/(np.sin(m)))

    return 90.-np.rad2deg(m), thea+np.rad2deg(np.where(easterly,g,-g))



def polar2latlonalt(r, az, elev, sitecoords, re=6370040.):
    """Transforms polar coordinates to lat/lon/altitude coordinates.

    Explicitely accounts for the beam's elevation angle and for the altitude of the radar location.

    This is an alternative implementation based on VisAD code (see
    http://www.ssec.wisc.edu/visad-docs/javadoc/visad/bom/Radar3DCoordinateSystem.html#toReference%28float[][]%29 and
    http://www.ssec.wisc.edu/~billh/visad.html ).

    VisAD code has been translated to Python from Java.

    Nomenclature tries to stick to VisAD code for the sake of comparibility, hwoever, names of
    arguments are the same as for polar2latlon...

    Parameters
    ----------
    r : array
        array of ranges [m]
    az : array
        array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of three floats
        the lat / lon coordinates of the radar location and its altitude a.m.s.l. (in meters)
        if sitecoords is of length two, altitude is assumed to be zero
    re : float
        earth's radius [m]

    Returns
    -------
    output : a tuple of three arrays (latitudes, longitudes, altitudes)

    """
    centlat = sitecoords[0]
    centlon = sitecoords[1]
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

    return lats, lons, alts


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

    Based on [Doviak1993]_ the beam height is calculated as

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

    References
    ----------
    .. [Doviak1993] Doviak R.J., Zrnic D.S, Doppler Radar and Weather
        Observations, Academic Press, 562pp, 1993, ISBN 0-12-221422-6
    """
    return np.sqrt( r**2 + (ke*re)**2 + 2*r*ke*re*np.sin(np.radians(theta)) ) - ke*re


def arc_distance_n(r, theta, re=6370040., ke=4./3.):
    r"""Calculates the great circle distance of a radar beam over a sphere,
    taking the refractivity of the atmosphere into account.

    Based on [Doviak1993]_ the arc distance may be calculated as

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
                             (ke*re + height_n(r, theta, re, ke)))


def polar2latlonalt_n(r, az, elev, sitecoords, re=6370040., ke=4./3.):
    """Transforms polar coordinates (of a PPI) to latitude/longitude \
    coordinates taking elevation angle and refractivity into account.

    This function assumes that the transformation from the polar radar
    coordinate system to the earth's spherical coordinate system may be done
    in the same way as astronomical observations are transformed from the
    horizon's coordinate system to the equatorial coordinate system.

    The conversion formulas used were taken from
    http://de.wikipedia.org/wiki/Nautisches_Dreieck [accessed 2001-11-02]

    It is based on polar2latlon but takes the shortening of the great circle
    distance by increasing the elevation angle as well as the resulting
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
    sitecoords : a sequence of two floats
        the lat / lon coordinates of the radar location
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    lat, lon, alt : tuple of arrays
        three arrays containing the spherical latitude and longitude coordinates
        as well as the altitude of the beam.

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
    >>> th = np.array([0.,   0.,   0.,   0.,   0.,  0.5,])
    >>> csite = (48.0, 9.0)
    >>> lat1, lon1, alt1 = polar2latlonalt(r, az, th, csite)
    >>> for x, y, z in zip(lat1, lon1, alt1):
    ...     print '{0:6.2f}, {1:6.2f}, {2:6.2f}'.format(x, y, z)
     48.00,   9.00,   0.00
     48.00,   9.00,   0.00
     49.00,   9.00, 725.30
     47.99,  10.49, 725.30
     47.00,   9.00, 725.30
     47.99,   7.51, 1693.81

    Here, the coordinates of the east and west directions won't come to lie on
    the latitude of the site because the beam doesn't travel along the latitude
    circle but along a great circle.


    """
    phi = np.deg2rad(sitecoords[0])
    lam = np.deg2rad(sitecoords[1])
    try:
        centalt = sitecoords[2]
    except:
        centalt = 0.

    # local earth radius
    re = re + centalt

    alt = beam_height_n(r, elev, re, ke)

    a   = np.deg2rad(-(180. + az))
    h   =  0.5*np.pi - arc_distance_n(r, elev, re, ke)/re

    delta, tau = hor2aeq(a, h, phi)
    latc = np.rad2deg(delta)
    lonc = np.rad2deg(lam + tau)

    return latc, lonc, alt


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
    polar2latlon.

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
        the lat / lon coordinates of the radar location

    Returns
    -------
    output : a 3-d array of polygon vertices in lon/lat
        with shape(num_vertices, num_vertex_nodes, 2). The last dimension
        carries the longitudes on the first position, the latitudes on the
        second (lon: output[:,:,0], lat: output[:,:,1]

    Examples
    --------
    >>> import numpy as pl
    >>> import pylab as pl
    >>> import matplotlib as mpl
    >>> # define the polar coordinates and the site coordinates in lat/lon
    >>> r = np.array([50., 100., 150., 200.])
    >>> az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])
    >>> sitecoords = (48.0, 9.0)
    >>> polygons = polar2polyvert(r, az, sitecoords)
    >>> # plot the resulting mesh
    >>> fig = pl.figure()
    >>> ax = fig.add_subplot(111)
    >>> polycoll = mpl.collections.PolyCollection(vertices,closed=True, facecolors=None)
    >>> ax.add_collection(polycoll, autolim=True)
    >>> pl.axis('tight')
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
    lat, lon= polar2latlon(r, az, sitecoords)

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

    For further information refer to the documentation of georef.polar2latlon.

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
        the lat / lon coordinates of the radar location
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
    lat, lon= polar2latlon(r, az, sitecoords)

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
        print 'Invalid polar coordinates: 0 is not a valid range gate specification (the centroid of a range gate must be positive).'
        exit()
    if len(np.unique(r))!=len(r):
        print 'Invalid polar coordinates: Range gate specification contains duplicate entries.'
        exit()
    if len(np.unique(az))!=len(az):
        print 'Invalid polar coordinates: Azimuth specification contains duplicate entries.'
        exit()
    if len(np.unique(az))!=len(az):
        print 'Invalid polar coordinates: Azimuth specification contains duplicate entries.'
        exit()
    if not _is_sorted(r):
        print 'Invalid polar coordinates: Range array must be sorted.'
        exit()
    if len(np.unique(r[1:]-r[:-1]))>1:
        print 'Invalid polar coordinates: Range gates are not equidistant.'
        exit()
    if len(np.where(az>=360.)[0])>0:
        print 'Invalid polar coordinates: Azimuth angles must not be greater than or equal to 360 deg.'
        exit()
    if not _is_sorted(az):
        # it is ok if the azimuth angle array is not sorted, but it has to be
        #   'continuously clockwise', e.g. it could start at 90° and stop at °89
        az_right = az[np.where(np.logical_and(az<=360, az>=az[0]))[0]]
        az_left = az[np.where(az<az[0])]
        if ( not _is_sorted(az_right) ) or ( not _is_sorted(az_left) ):
            print 'Invalid polar coordinates: Azimuth array is not sorted clockwise.'
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
    >>> # UTM zone 51 (northern hemisphere)
    >>> print create_projstr("utm", zone=51)
    >>> # UTM zone 51 (southern hemisphere)
    >>> print create_projstr("utm", zone=51, hemisphere="south")

    """
    if projname=="aeqd":
        # Azimuthal Equidistant
        projstr = """+proj=aeqd  +lat_0=%s +lon_0=%s +x_0=%s +y_0=%s""" \
                  % (kwargs["lat_0"], kwargs["lon_0"], kwargs["x_0"], kwargs["y_0"])
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
        try:
            projstr = "+proj=utm +zone=%d +ellps=WGS84%s" % (kwargs["zone"], hemisphere)
        except:
            print "Cannot create projection string for projname %s. Maybe keyword argument zone was not passed?" % s
            exit(1)

    elif projname=="dwd-radolan":
        # DWD-RADOLAN polar stereographic projection
        scale = (1.+np.sin(np.radians(60.)))/(1.+np.sin(np.radians(90.)))
        projstr = ('+proj=stere +lat_0=90 +lon_0=10 +k_0={0:10.8f} '+
                   '+ellps=sphere +a=6370040.000 +es=0.0').format(scale)
    else:
        print "No support for projection %r, yet." % projname
        print "You need to create projection string by hand..."
        exit(1)
    return projstr


def project(latc, lonc, projstr, inverse=False):
    """
    Convert from latitude,longitude (based on WGS84) to coordinates in map projection

    This mainly serves as a convenience function to use proj.4 via pyproj.
    For proj.4 documentation visit http://proj.maptools.org.
    For pyproj documentation visit http://code.google.com/p/pyproj.

    See http://www.remotesensing.org/geotiff/proj_list for examples of key/value
    pairs defining different map projections.

    You can use :doc:`wradlib.georef.create_projstr` in order to create projection
    strings to be passed with argument *projstr*. However, the choice is still
    rather limited. Alternatively, you have to create or look up projection strings by yourself.

    See the Examples section for a quick start.

    Parameters
    ----------
    latc : array of floats
        latitude coordinates based on WGS84
    lonc : array of floats
        longitude coordinates based on WGS84
    projstr : string
        proj.4 projection string. Can be conveniently created by using function
        :doc:`wradlib.georef.create_projstr`

    Returns
    -------
    output : a tuple of 2 arrays (x and y coordinates)

    Examples
    --------
    Gauss-Krueger Zone 2:
        *"+proj=tmerc +lat_0=0 +lon_0=6 +k=1 +x_0=2500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"*

    Gauss-Krueger Zone 3:
        *"+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"*

    UTM Zone 51 on the Northern Hemishpere
        *"+proj=utm +zone=51 +ellps=WGS84"*

    UTM Zone 51 on the Southern Hemishpere
        *"+proj=utm +zone=51 +ellps=WGS84 +south"*

    >>> import wradlib.georef as georef
    >>> # This is Gauss-Krueger Zone 3 (aka DHDN 3 aka Germany Zone 3)
    >>> gk3 = create_projstr("gk", zone=3)
    >>> latc = [54.5, 55.5]
    >>> lonc = [9.5, 9.8]
    >>> gk3_x, gk3_y = georef.project(latc, lonc, gk3)

    """
    myproj = pyproj.Proj(projstr)
    x, y = myproj(lonc, latc, inverse=inverse)
    return x, y


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
    x, y = project(cent_lat, cent_lon, projstr)
    return x.ravel(), y.ravel()


def _doctest_():
    import doctest
    print 'doctesting'
    doctest.testmod()
    print 'finished'


if __name__ == '__main__':
    print 'wradlib: Calling module <georef> as main...'
