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

import wradlib.ipol as ipol

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


def togrid(src, trg, radius, center, data, interpol, *args, **kwargs):
    """
    Interpolate data from a radar location to the composite grid or set of locations

    Parameters
    ----------
    src : ndarray of float of shape (numpoints, ndim)
        cartesian x / y coordinates of the radar bins
    trg : ndarray of float of shape (numpoints, ndim)
        cartesian x / y coordinates of the composite
    radius : float - the radius of the radar circle (same units as src and trg)
    center : array of float - the location coordinates of the radar
    data : ndarray of float - the data that should be transferred to composite
    interpol : an interpolation class name from wradlib.ipol

    Returns
    -------
    output : ndarray of float
        data of the radar circle which is interpolated on the composite grid

    """
    # get indices to select the subgrid from the composite grid
    ix = extract_circle(center, radius, trg)
    # interpolate on subgrid
    ip = interpol(src, trg[ix], *args, **kwargs)
    data_on_subgrid = ip(data).reshape( (len(ix),-1) )
    # create container for entire grid
    compose_grid = np.repeat( np.nan, len(trg)*data.ndim ).reshape( (len(trg), data.ndim) )
    # push subgrid results into the large grid
    compose_grid[ix] = data_on_subgrid
    return compose_grid




if __name__ == '__main__':
    print 'wradlib: Calling module <comp> as main...'
    import numpy as np
    import datetime as dt
    import pylab as pl


    radarcoords = np.loadtxt('../examples/data/bin_coords_tur.gz')
    center = radarcoords.mean(axis=0)

    coords = np.meshgrid(np.linspace(center[0]-3e5,center[0]+3e5,900), np.linspace(center[1]-3e5,center[1]+3e5,900))
    coords = np.vstack((coords[0].ravel(), coords[1].ravel())).transpose()

    #center = np.array([500.,500.])
    radius = 128000.

    radardata = np.loadtxt('../examples/data/polar_R_tur.gz').ravel()
##    t0 = dt.datetime.now()
##    ix = extract_circle(center, radius, coords)
##    print dt.datetime.now() - t0

    t0 = dt.datetime.now()
    test = togrid(radarcoords, coords, radius, center, radardata, ipol.Nearest)
    print dt.datetime.now() - t0

    pl.imshow(test.reshape((900,900)))

##    print len(ix)
##    print ix

##    pl.scatter(coords[ix,0], coords[ix,1])
    pl.show()
    pl.close()



