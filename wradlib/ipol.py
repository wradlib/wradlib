#-------------------------------------------------------------------------------
# Name:        ipol
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
Interpolation
^^^^^^^^^^^^^

Interpolation allows to transfer data from one set of locations to another.
This includes for example:

- interpolating the data from a polar grid to a cartesian grid or irregular points

- interpolating point observations to a grid or a set of irregular points

- filling missing values, e.g. filling clutters

.. autosummary::
   :nosignatures:
   :toctree: generated/

   Nearest
   Idw
   Linear
   interpolate

"""

from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import wradlib.util as util


class IpolBase():
    """
    IpolBase(src, trg)

    The base class for interpolation in N dimensions.
    Provides the basic interface for all other classes.

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.

    """

    def __init__(self, src, trg):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsources, ...)
            Values at the source points which to interpolate

        Returns
        -------
        output : None

        """
        self._check_shape(vals)
        return None
    def _check_shape(self, vals):
        """
        Checks whether the values correspond to the source points

        Parameters
        ----------
        vals : ndarray of float

        """
        assert len(vals)==self.numsources, 'Length of value array %d does not correspond to \
        number of source points %d' % (len(vals), self.numsources)
    def _make_coord_arrays(self, x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)

        Parameters
        ----------
        x : ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and the length
            of the ndarray corresponding to the number of points

        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif type(x)==np.ndarray:
            if x.ndim==1:
                x = x.reshape(-1,1)
            elif x.ndim==2:
                pass
            else:
                raise Exception('Cannot deal wih 3-d arrays, yet.')
        return x



class Nearest(IpolBase):
    """
    Nearest(src, trg)

    Nearest-neighbour interpolation in N dimensions.

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    """

    def __init__(self, src, trg):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        self.numsources = len(src)
        # plant a tree
        self.tree = cKDTree(src)
        self.dists, self.ix = self.tree.query(trg, k=1)
    def __call__(self, vals, maxdist=None):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        maxdist : the maximum distance up to which an interpolated values is
            assigned - if maxdist is exceeded, np.nan will be assigned
            If maxdist==None, values will be assigned everywhere

        Returns
        -------
        output : ndarray of float with shape (numtargetpoints,...)

        """
        self._check_shape(vals)
        out = vals[self.ix]
        if maxdist==None:
            return out
        else:
            return np.where(self.dists>maxdist, np.nan, out)


class Idw(IpolBase):
    """
    Idw(src, trg, nnearest=4, p=2.)

    Inverse distance weighting interpolation in N dimensions.

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    nnearest : integer - max. number of neighbours to be considered
    p : float - inverse distance power used in 1/dist**p

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    """

    def __init__(self, src, trg, nnearest=4, p=2.):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        self.numsources = len(src)
        self.nnearest = nnearest
        self.p = p
        # plant a tree
        self.tree = cKDTree(src)
        self.dists, self.ix = self.tree.query(trg, k=nnearest)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:,np.newaxis]
            self.ix = self.ix[:,np.newaxis]


    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        maxdist : the maximum distance up to which an interpolated values is
            assigned - if maxdist is exceeded, np.nan will be assigned
            If maxdist==None, values will be assigned everywhere

        Returns
        -------
        output : ndarray of float with shape (numtargetpoints,...)

        """

        # self distances: a list of arrays of distances of the nearest points which are indicated by self.ix
        outshape = list( vals.shape )
        outshape[0] = len(self.dists)
        interpol = np.repeat(np.nan, util._shape2size(outshape)).reshape(tuple(outshape)).astype('f4')
        # weights is the container for the weights (a list)
        weights  = range( len(self.dists) )
        # sources is the container for the source point indices
        src_ix   = range( len(self.dists) )
        # jinterpol is the jth element of interpol
        jinterpol = 0
        for dist, ix in zip( self.dists, self.ix ):
            valid_dists = np.where(np.isfinite(dist))[0]
            dist = dist[valid_dists]
            ix = ix[valid_dists]
            if self.nnearest == 1:
                # defaults to nearest neighbour
                wz = vals[ix]
                w = 1.
            elif dist[0] < 1e-10:
                # if a target point coincides with a source point
                wz = vals[ix[0]]
                w  = 1.
            else:
                # weight z values by (1/dist)**p --
                w = 1. / dist**self.p
                w /= np.sum(w)
                wz = np.dot( w, vals[ix] )
            interpol[jinterpol] = wz.ravel()
            weights [jinterpol] = w
            src_ix  [jinterpol] = ix
            jinterpol += 1
        return interpol ## if self.qdim > 1  else interpol[0]


class Linear(IpolBase):
    """
    Interface to the scipy.interpol.LinearNDInterpolator class.

    We provide this class in order to achieve a uniform interface for all
    Interpolator classes

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.

    """

    def __init__(self, src, trg):
        self.src = self._make_coord_arrays(src)
        self.trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        self.numsources = len(src)
    def __call__(self, vals, fill_value=np.nan):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        fill_value : float
            is needed if linear interpolation fails; defaults to np.nan

        Returns
        -------
        output : ndarray of float with shape (numtargetpoints,...)

        """
        self._check_shape(vals)
        ip = LinearNDInterpolator(self.src, vals, fill_value=fill_value)
        return ip(self.trg)

def interpolate(src, trg, vals, Interpolator, *args, **kwargs):
    """
    Convenience function to use the interpolation classes in an efficient way

    ATTENTION: Works only for one- and two-dimensional *vals* arrays, yet.

    The interpolation classes in wradlib.ipol are computationally very efficient
    if they are applied on large multi-dimensional arrays of which the first dimension
    must be the locations' dimension (1d or 2d coordinates) and the following dimensions
    can be anything (e.g. time or ensembles). This way, the weights need to be computed
    only once. However, this can only be done with success if all source values for
    the interpolation are valid numbers. If the source values contain let's say
    *np.nan* types, the result of the interpolation will be *np.nan* in the
    vicinity of the corresponding points, too. Imagine that you have a time series
    of observations at points and in each time step one observation is missing.
    You would still like to efficiently apply the interpolation
    classes, but you will need to account for the resulting *np.nan* values in
    the interpolation output.

    In order to still allow for the efficient application, you have to take care
    of the remaining np.nan in your interpolation result. This is done by this
    convenience function.

    Alternatively, you have to make sure that your *vals* argument does not contain
    any *np.nan* values OR you have to post-process missing values in your interpolation
    result in another way.

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    vals : ndarray of float, shape (numsourcepoints, ...)
        Values at the source points which to interpolate
    Interpolator : a class which inherits from IpolBase
    *args : arguments of Interpolator (see class documentation)
    **kwargs : keyword arguments of Interpolator (see class documentation)

    Examples
    --------
    >>> # test for 1 dimension in space and two value dimensions
    >>> src = np.arange(10)[:,None]
    >>> trg = np.linspace(0,20,40)[:,None]
    >>> vals = np.hstack((np.sin(src), 10.+np.sin(src)))
    >>> # here we introduce missing values only in the second dimension
    >>> vals[3:5,1] = np.nan
    >>> ipol_result = interpolate(src, trg, vals, Idw, nnearest=2)
    >>> # plot if you like
    >>> import pylab as pl
    >>> pl.plot(trg, ipol_result, 'b+')
    >>> pl.plot(src, vals, 'ro')
    >>> pl.show()


    """
    if vals.ndim==1:
        # source values are one dimensional, we have just to remove invalid data
        ix_valid = np.where(np.isfinite(vals))[0]
        ip = Interpolator(src[ix_valid], trg, *args, **kwargs)
        result = ip(vals[ix_valid])
    elif vals.ndim==2:
        # this implementation for 2 dimensions needs generalization
        ip = Interpolator(src, trg, *args, **kwargs)
        result = ip(vals)
        nan_in_result = np.where(np.isnan(result))
        nan_in_vals = np.where(np.isnan(vals))
        for i in np.unique(nan_in_result[-1]):
            ix_good = np.where(np.isfinite(vals[...,i]))[0]
            ix_broken_targets = nan_in_result[0][np.where(nan_in_result[-1]==i)[0]]
            ip = Interpolator(src[ix_good], trg[nan_in_result[0][np.where(nan_in_result[-1]==i)[0]]], *args, **kwargs)
            tmp = ip(vals[ix_good,i].reshape((len(ix_good),-1)))
            result[ix_broken_targets,i] = tmp.ravel()
    else:
        if not np.any(np.isnan(vals.ravel())):
            raise Exception('At the moment, <interpolate> can only deal with NaN values in <vals> if vals has less than 3 dimension.')
        else:
            # if no NaN value are in <vals> we can safely apply the Interpolator as is
            ip = Interpolator(src, trg, *args, **kwargs)
            result = ip(vals[ix_valid])
    return result


if __name__ == '__main__':
    print 'wradlib: Calling module <ipol> as main...'




