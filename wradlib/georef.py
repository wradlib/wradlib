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
   project

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


def hor2aeq(a, h, phi):
    """"""
    delta = arcsin(-cos(h)*cos(a)*cos(phi) + sin(h)*sin(phi))
    tau = arcsin(cos(h)*sin(a)/cos(delta))
    return delta, tau


def aeq2hor(tau, delta, phi):
    """"""
    h = arcsin(cos(delta)*cos(tau)*cos(phi) + sin(delta)*sin(phi))
    a = arcsin(cos(delta)*sin(tau)/cos(h))


def polar2latlon(r, az, sitecoords, re=6370.04):
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
        array of ranges [km]
    az : array
        array of azimuth angles containing values between 0° and 360°.
        These are assumed to start with 0° pointing north and counted positive
        clockwise!
    sitecoords : a sequence of two floats
        the lat / lon coordinates of the radar location
    re : float
        earth's radius [km]

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


def __pol2latlon(rng, az, sitecoords, re=6370.04):
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


def centroid2polyvert(centroid, delta):
    """Calculates the 2-D Polygon vertices necessary to form a rectangular
    Polygon around the centroid's coordinates.

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


def polar2polyvert(r, az, sitecoords, re=6370.04):
    """
    Generate 2-D polygon vertices directly from polar coordinates

    This is an alternative to centroid2polyvert which does not use centroids,
    but generates the polygon vertices by simply connecting the corners of the
    radar bins.

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. For further information refer to the documentation of
    polar2latlon.

    r : array
        array of ranges [km]; r defines the exterior boundaries of the range bins!
        (not the centroids). Thus, values must be positive!
    az : array
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main beam lobe!
        The first angle can start at any values, but make sure the array is sorted
        continuously positively clockwise and the angles are equidistant. An angle
        if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lat / lon coordinates of the radar location
    re : float
        earth's radius [km]

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
    >>> r = np.array([0., 50., 100., 150.])
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




def polar2centroids(r=None, az=None, sitecoords=None, re=6370.04):
    """
    Computes the centroids of the radar bins from their polygon vertices

    Both azimuth and range arrays are assumed to be equidistant and to contain
    only unique values. The ranges are assumed to define the exterior boundaries
    of the range bins (thus they must be positive). The angles are assumed to
    describe the pointing direction fo the main beam lobe.

    For further information refer to the documentation of georef.polar2latlon.

    r : array
        array of ranges [km]; r defines the exterior boundaries of the range bins!
        (not the centroids). Thus, values must be positive!
    az : array
        array of azimuth angles containing values between 0° and 360°.
        The angles are assumed to describe the pointing direction fo the main beam lobe!
        The first angle can start at any values, but make sure the array is sorted
        continuously positively clockwise and the angles are equidistant. An angle
        if 0 degree is pointing north.
    sitecoords : a sequence of two floats
        the lat / lon coordinates of the radar location
    re : float
        earth's radius [km]

    Returns
    -------
    output : tuple of 2 arrays which describe the bin centroids
        longitude and latitude

    Note
    ----
    Azimuth angles of 360 deg are internally converted to 0 deg.

    """
    # make sure the range and azimuth angles have the right properties
    r, az = _check_polar_coords(r, az)

    # to get the centroid, we only have to move the exterior bin boundaries by half the resolution
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
        print 'Invalid polar coordinates: Azimuth angles are not equidistant.'
        exit()
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


def project(latc, lonc, projstr):
    """
    Convert from latitude,longitude (based on WGS84)
    to coordinates in map projection

    This mainly serves as a convenience function to use proj.4 via pyproj.
    For proj.4 documentation visit http://proj.maptools.org.
    For pyproj documentation visit http://code.google.com/p/pyproj.

    See http://www.remotesensing.org/geotiff/proj_list for examples of key/value
    pairs defining different map projections.

    The main challenge is to formulate an appropriate proj.4 projection string
    for the target projection. See the Examples section for a quick start.

    Parameters
    ----------
    latc : array of floats
        latitude coordinates based on WGS84
    lonc : array of floats
        longitude coordinates based on WGS84
    projstr : string
        proj.4 projection string

    Returns
    -------
    output : a tuple of 2 arrays (x and y coordinates)

    Examples
    --------
    Gauss-Krueger Zone 2:
        "+proj=tmerc +lat_0=0 +lon_0=6 +k=1 +x_0=2500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"

    Gauss-Krueger Zone 3:
        "+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"

    UTM Zone 51 on the Northern Hemishpere
        "+proj=utm +zone=51 +ellps=WGS84"

    UTM Zone 51 on the Southern Hemishpere
        "+proj=utm +zone=51 +ellps=WGS84 +south"

    >>> import wradlib.georef as georef
    >>> # This is Gauss-Krueger Zone 3 (aka DHDN 3 aka Germany Zone 3)
    >>> gk3 = '''
    >>> +proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
    >>> +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs
    >>> '''
    >>> latc = [54.5, 55.5]
    >>> lonc = [9.5, 9.8]
    >>> gk3_x, gk3_y = georef.project(latc, lonc, gk3)

    """
    myproj = pyproj.Proj(projstr)
    x, y = myproj(lonc, latc)
    return x, y


def projected_bincoords_from_radarspecs(r, az, sitecoords, projstr):
    """
    Convenience function to compute projected bin coordinates directly from
    radar site coordinates and range/azimuth specs

    Parameters
    ----------
    r : array
    az : array
    sitecoords : tuple
    projstr : string

    """
    cent_lon, cent_lat = polar2centroids(r, az, sitecoords)
    return project(cent_lat, cent_lon, projstr)


def _doctest_():
    import doctest
    print 'doctesting'
    doctest.testmod()
    print 'finished'


if __name__ == '__main__':
    print 'wradlib: Calling module <georef> as main...'
#    _doctest_()

