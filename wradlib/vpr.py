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

*UNDER DEVELOPMENT*

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

   volcoords_from_polar
   CartesianVolume

"""

import numpy as np
import wradlib.georef as georef
import wradlib.ipol as ipol
import wradlib.util as util
import wradlib.io as io
from scipy.spatial import cKDTree
import cPickle as pickle
import os


class CartesianVolume():
    """Create 3-D regular volume grid in Cartesian coordinates from polar data with multiple elevation angles

    Parameters
    ----------
    polcoords : array of shape (number of bins, 3)
    cartcoords : array of shape (number of voxels, 3)
    polshape : shape of the original volume (num elevation angles, num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maskfile : path to an hdf5 file (default: empty string)
        File should contain a boolean array which masks the "blind" areas of the volume scan
    Ipclass : an interpolation class from wradlib.ipol
    ipargs : keyword arguments corresponding to Ipclass

    Returns
    -------
    output : float ndarray of shape (number of levels, number of x coordinates, number of y coordinates)

    """
    def __init__(self, polcoords, cartcoords, polshape, maxrange, maskfile="", Ipclass=ipol.Idw, **ipargs):
        self.Ipclass        = Ipclass
        self.ipargs         = ipargs
        # create a default instance of interpolator
        print "Creating 3D interpolator...this is still very slow."
        self.ip             = Ipclass(src=polcoords, trg=cartcoords, **ipargs)
        try:
            # read mask from pickled file
            self.mask = io.from_hdf5(maskfile)[0]
            # check whether mask is consistent with the data
            if not len(self.mask)==len(cartcoords):
                raise Exception()
            print "Load mask from file <%s>: successful" % maskfile
        except:
            self.mask = self.create_mask(polcoords, cartcoords, polshape, maxrange)
            try:
                io.to_hdf5(maskfile, self.mask, dtype="bool")
                print "Save mask to file <%s>: successful" % maskfile
            except:
                pass

    def __call__(self, data):
        """Interpolates the polar data to 3-dimensional Cartesian coordinates

        Parameters
        ----------
        data : 1-d array of length (num voxels,)

        """
        ipdata = self.ip(data)
        ipdata[self.mask] = np.nan
        return ipdata

    def create_mask(self, polcoords, cartcoords, polshape, maxrange):
        """Identifies all the "blind" voxels of a Cartesian 3D-volume grid
        """
        print "Creating volume mask from scratch...this is still very slow."
        # Identify voxels beyond the maximum range
        center = np.array([np.mean(polcoords[:,0]), np.mean(polcoords[:,1]), np.min(polcoords[:,2])]).reshape((-1,3))
        in_range = ((cartcoords-center)**2).sum(axis=-1) <= maxrange**2
        # Identify those grid altitudes above the maximum scanning angle
        maxelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[-1].ravel(),polcoords[:,1].reshape(polshape)[-1].ravel(),polcoords[:,2].reshape(polshape)[-1].ravel())).transpose()
        alt_interpolator = ipol.Nearest(maxelevcoords, cartcoords)
        maxalt = alt_interpolator(maxelevcoords[:,2])
        # Identify those grid altitudes below the minimum scanning angle
        minelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[0].ravel(),polcoords[:,1].reshape(polshape)[0].ravel(),polcoords[:,2].reshape(polshape)[0].ravel())).transpose()
        alt_interpolator = ipol.Nearest(minelevcoords, cartcoords)
        minalt = alt_interpolator(minelevcoords[:,2])
        # mask those values above the maximum and below the minimum scanning angle
        return np.logical_not( np.logical_and(np.logical_and(cartcoords[:,2]<=maxalt, cartcoords[:,2]>=minalt), in_range) )


def volcoords_from_polar(sitecoords, elevs, azimuths, ranges, projstr=None):
    """Create Cartesian coordinates for the polar volume bins

    Parameters
    ----------
    sitecoords : sequence of three floats indicating the radar position
       (latitude in decimal degrees, longitude in decimal degrees, height a.s.l. in meters)
    elevs : sequence of elevation angles
    azimuths : sequence of azimuth angles
    ranges : sequence of ranges
    projstr : proj.4 projection string

    Returns
    -------
    output : array of shape (num volume bins, 3)

    """
    # create polar grid
    el, az, r = util.meshgridN(elevs, azimuths, ranges)
    # get geographical coordinates
    lats, lons, z = georef.polar2latlonalt(r, az, el, sitecoords, re=6370040.)
    # get projected horizontal coordinates
    x, y = georef.project(lats, lons, projstr)
    # create standard shape
    coords = np.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return coords


if __name__ == '__main__':
    print 'wradlib: Calling module <vpr> as main...'


