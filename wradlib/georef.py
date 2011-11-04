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




if __name__ == '__main__':
    print 'wradlib: Calling module <georef> as main...'
    r = np.array([0.,0.,10.,10.,10.,10.,])
    az = np.array([0.,180.,0.,90.,180.,270.,])
    csite = (48.0, 9.0)
    print polar2latlon(r, az, csite)
