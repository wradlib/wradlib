#-------------------------------------------------------------------------------
# Name:        verify
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
Verification
^^^^^^^^^^^^

Verification mainly refers to the comparison of radar-based precipitation
estimates to ground truth.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   PolarNeighbours

"""

import numpy as np
from scipy.spatial import KDTree

import georef

class PolarNeighbours():
    """
    For a set of projected point coordinates, extract the neighbouring bin values
    from a data set in polar coordinates

    Parameters
    ----------
    r : array of floats
    az : array of floats
    sitecoords : sequence of floats
    projstr : string
    x : array of floats
    y : array of floats
    nnear : int

    """
    def __init__(self, r, az, sitecoords, projstr, x, y, nnear=9):
        self.nnear = nnear
        self.az = az
        self.r = r
        self.x = x
        self.y = y
        # compute the centroid coordinates in lat/lon
        bin_lon, bin_lat = georef.polar2centroids(r, az, sitecoords)
        # project the centroids to cartesian map coordinates
        binx, biny = georef.project(bin_lat, bin_lon, projstr)
        self.binx, self.biny = binx.ravel(), biny.ravel()
        # compute the KDTree
        tree = KDTree(zip(self.binx, self.biny))
        # query the tree for nearest neighbours
        self.dist, self.ix = tree.query(zip(x, y), k=nnear)
    def extract(self, vals):
        """
        Extracts the values from an array of shape (azimuth angles, range gages)
        which correspond to the indices computed during initialisation

        Parameters
        ----------
        vals : array of shape (..., number of azimuth, number of range gates)

        Returns
        -------
        output : array of shape (..., number of points, nnear)

        """
        assert vals.ndim >= 2, \
           'Your <vals> array should at least contain an azimuth and a range dimension.'
        assert tuple(vals.shape[-2:])==(len(self.az), len(self.r)), \
           'The shape of your vals array does not correspond with the range and azimuths you provided for your polar data set'
        shape = vals.shape
        vals = vals.reshape(np.concatenate( (shape[:-2], np.array([len(self.az) * len(self.r)])) ) )
        return vals[...,self.ix]
    def get_bincoords(self):
        return self.binx, self.biny
    def get_bincoords_at_points(self):
        return self.binx[self.ix], self.biny[self.ix]





if __name__ == '__main__':
    print 'wradlib: Calling module <verify> as main...'
