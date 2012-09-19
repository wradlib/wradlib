#-------------------------------------------------------------------------------
# Name:        vpr
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
Vertical Profile of Reflectivity (VPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UNDER DEVELOPMENT

Precipitation is 3-dimensional in space. The vertical distribution of precipitation
(and thus reflectivity) is typically non-uniform. As the height of the radar beam increases with the distance
from the radar location (beam elevation, earth curvature), the one sweep samples
from different heights. The effects of the non-uniform VPR and the differnt sampling heights need
to be accounted for if we are interested in the precipiation near the ground or
in defined heights. This module is intended to provide a set of tools to account
for these effects.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   cappi

"""

def cappi(data, sitecoords, elevs, levels, dx, dy, maxvert, maxhoriz, projstr):
    """UNDER DEVELOPMENT: Create a CAPPI from sweep data with multiple elevation angles

    Parameters
    ----------
    data : float ndarray with shape (num elevations, num azimuth angles, num range bins)
    sitecoords : sequence of three floats indicating the radar position
       (latitude in decimal degrees, longitude in decimal degrees, height a.s.l. in meters)
    elevs : sequence of elevation angles corresponding to first dimension of data
    levels : sequence of floats
       target altitudes for CAPPI (in meters)
    dx : float
       horizontal resolution of CAPPI in x direction
    dy : float
       horizontal resolution of CAPPI in y direction
    maxvert : float
       maximum vertical distance threshold - the next data bin must be closer to
       the target location than maxvert in order to assign a value
    maxhoriz : float
       maximum horizontal distance threshold - the next data bin must be closer to
       the target location than maxhoriz in order to assign a value
    projstr : proj.4 projection string

    Returns
    -------
    output : float ndarray of shape (number of levels, number of x coordinates, number of y coordinates)

    """
    pass



if __name__ == '__main__':
    print 'wradlib: Calling module <vpr> as main...'
