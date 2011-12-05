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
import pyproj as proj


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


def project(latc, lonc, projstr):
    """
    Convert from latitude,longitude (based on WGS84)
    to coordinates in map projection

    This mainly serves as a convenience function to use proj.4 via pyproj.
    For proj.4 documentation visit http://proj.maptools.org. For pyproj
    documentation visit http://code.google.com/p/pyproj. See
    http://www.remotesensing.org/geotiff/proj_list for examples of  key/value
    pairs defining different map projections.

    The main challenge is to formulate an appropriate proj.4 projection string
    for the target projection. Examples are given in the parameters section.

    Parameters
    ----------
    latc : array of floats
        latitude coordinates based on WGS84
    lonc : array of floats
        longitude coordinates based on WGS84
    projstr : string
        proj.4 projection string

    Examples
    --------
    Gauss-Krueger Zone 2:
        "+proj=tmerc +lat_0=0 +lon_0=6 +k=1 +x_0=2500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"

    Gauss-Krueger Zone 3:
        "+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
        +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs"

    >>> import wradlib.georef as georef
    >>> gk3 = '''
    >>> +proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
    >>> +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs
    >>> '''
    >>> latc = [54.5, 55.5]
    >>> lonc = [9.5, 9.8]
    >>> gk3_coords = georef.project(latc, lonc, gk3)

    """
    myproj = proj.Proj(projstr)
    mapcoords = myproj(lonc, latc)
    return mapcoords





if __name__ == '__main__':
    print 'wradlib: Calling module <georef> as main...'
##    r = np.array([0.,0.,10.,10.,10.,10.,])
##    az = np.array([0.,180.,0.,90.,180.,270.,])
##    csite = (48.0, 9.0)
##    print polar2latlon(r, az, csite)

    latlon = np.genfromtxt('E:/test/georeftest/latlon.txt', names=True)
    projected = project(projstr="", latc=latlon['y'], lonc=latlon['x'])
    f = open('E:/test/georeftest/projected.txt', 'w')
    f.write('x\ty\n')
    np.savetxt(f, np.transpose(projected), delimiter='\t', fmt='%.2f')
    f.close()

