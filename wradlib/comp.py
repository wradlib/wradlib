#-------------------------------------------------------------------------------
# Name:        comp
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
Composition
^^^^^^^^^^^

Combine data from different radar locations on one common set of locations

.. autosummary::
   :nosignatures:
   :toctree: generated/

   extract_circle

"""

##from scipy.spatial import KDTree
##def extract_circle(center, radius, coords):
##    """
##    Extract the indices of coords which fall within a circle
##    defined by center and radius
##
##    Parameters
##    ----------
##    center : float
##    radius : float
##    coords : array of float with shape (numpoints,2)
##
##    Returns
##    -------
##    output : 1-darray of integers
##        index array referring to the coords array
##
##    """
##    print 'Building tree takes:'
##    t0 = dt.datetime.now()
##    tree = KDTree(coords)
##    print dt.datetime.now() - t0
##    print 'Query tree takes:'
##    t0 = dt.datetime.now()
##    ix = tree.query(center, k=len(coords), distance_upper_bound=radius)[1]
##    print dt.datetime.now() - t0
##    ix = ix[np.where(ix<len(coords))[0]]
##    return ix

def extract_circle(center, radius, coords):
    """
    Extract the indices of coords which fall within a circle
    defined by center and radius

    Parameters
    ----------
    center : float
    radius : float
    coords : array of float with shape (numpoints,2)

    Returns
    -------
    output : 1-darray of integers
        index array referring to the coords array

    """
    return np.where( ((coords-center)**2).sum(axis=1) < radius**2 )[0]



if __name__ == '__main__':
    print 'wradlib: Calling module <comp> as main...'
##    import numpy as np
##    import datetime as dt
##    import pylab as pl
##
##    coords = np.meshgrid(np.arange(900), np.arange(900))
##    coords = np.vstack((coords[0].ravel(), coords[1].ravel())).transpose()
##    center = np.array([500.,500.])
##    radius = 128.
##    t0 = dt.datetime.now()
##    ix = extract_circle(center, radius, coords)
##    print dt.datetime.now() - t0
##
##    print len(ix)
##    print ix
##
##    pl.scatter(coords[ix,0], coords[ix,1])
##    pl.show()
##    pl.close()



