#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Verification
^^^^^^^^^^^^

Verification mainly refers to the comparison of radar-based precipitation
estimates to ground truth.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["ErrorMetrics", "PolarNeighbours"]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from pprint import pprint

import numpy as np
from scipy import spatial, stats

from wradlib import util
from wradlib.georef import polar


class PolarNeighbours:
    """For a set of projected point coordinates, extract the neighbouring bin \
    values from a data set in polar coordinates.

    Use as follows:

    First, create an instance of PolarNeighbours by passing all the information
    needed to georeference the polar radar data to the points of interest
    (see parameters).

    Second, use the method *extract* in order to extract the values from a data
    array which corresponds to the polar coordinates.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        (see georef for documentation)
    az : :class:`numpy:numpy.ndarray`
        (see georef for documentation)
    sitecoords : sequence of floats
        (see georef for documentation)
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection
        (see georef for documentation)
    x : :class:`numpy:numpy.ndarray`
        array of x coordinates of the points in map projection
        corresponding to proj
    y : :class:`numpy:numpy.ndarray`
        array of y coordinates of the points in map projection
        corresponding to proj
    nnear : int
        number of neighbouring radar bins you would like to find

    Examples
    --------

    See :ref:`/notebooks/verification/wradlib_verify_example.ipynb`.

    """

    def __init__(self, r, az, sitecoords, proj, x, y, nnear=9):
        self.nnear = nnear
        self.az = az
        self.r = r
        self.x = x
        self.y = y
        # compute the centroid coordinates in proj
        bin_coords = polar.spherical_to_centroids(r, az, 0, sitecoords, proj=proj)
        self.binx = bin_coords[..., 0].ravel()
        self.biny = bin_coords[..., 1].ravel()
        # compute the KDTree
        tree = spatial.KDTree(list(zip(self.binx, self.biny)))
        # query the tree for nearest neighbours
        self.dist, self.ix = tree.query(list(zip(x, y)), k=nnear)

    def extract(self, vals):
        """Extracts the values from an array of shape (azimuth angles, \
        range gages) which correspond to the indices computed during \
        initialisation

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            array of shape (..., number of azimuth, number of range gates)

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            array of shape (..., number of points, nnear)

        """
        assert vals.ndim >= 2, (
            "Your <vals> array should at least contain an "
            "azimuth and a range dimension."
        )
        assert tuple(vals.shape[-2:]) == (len(self.az), len(self.r)), (
            "The shape of your vals array does not correspond with "
            "the range and azimuths you provided for your polar data set"
        )
        vals = vals.reshape(vals.shape[:-2] + (len(self.az) * len(self.r),))
        return vals[..., self.ix]

    def get_bincoords(self):
        """Returns all bin coordinates in map projection

        Returns
        -------
        output : tuple
            array of x coordinates, array of y coordinates

        """
        return self.binx, self.biny

    def get_bincoords_at_points(self):
        """Returns bin coordinates only in the neighbourhood of points

        Returns
        -------
        output : tuple
            array of x coordinates, array of y coordinates

        """
        return self.binx[self.ix], self.biny[self.ix]


class ErrorMetrics:
    """Compute quality metrics from a set of observations (``obs``) and \
    estimates (``est``).

    First create an instance of the class using the set of observations and
    estimates. Then compute quality metrics using the class methods.
    A dictionary of all available quality metrics is returned using the
    ``all`` method, or printed to the screen using the ``pprint`` method.

    The ``ix`` member variable indicates valid pairs of ``obs`` and ``est``,
    based on NaNs and ``minval``.

    Parameters
    ----------
    obs: :class:`numpy:numpy.ndarray`
        array of observations (e.g. rain gage observations)
    est: :class:`numpy:numpy.ndarray`
        array of estimates (e.g. radar, adjusted radar, ...)
    minval : float
        threshold value in order to compute metrics only for values larger
        than minval

    Examples
    --------
    >>> obs = np.random.uniform(0, 10, 100)
    >>> est = np.random.uniform(0, 10, 100)
    >>> metrics = ErrorMetrics(obs, est)
    >>> metrics.all() #doctest: +SKIP
    >>> metrics.pprint() #doctest: +SKIP
    >>> metrics.ix #doctest: +SKIP

    See :ref:`/notebooks/verification/wradlib_verify_example.ipynb` and
    :ref:`/notebooks/multisensor/wradlib_adjust_example.ipynb`.

    """

    def __init__(self, obs, est, minval=None):
        # Check input
        if len(obs) != len(est):
            raise ValueError(
                "WRADLIB: obs and est need to have the "
                "same length. len(obs)={}, "
                "len(est)={}".format(len(obs), len(est))
            )
        self.est = est
        self.obs = obs
        # remember those pairs which both have valid obs and est
        self.ix = np.intersect1d(
            util._idvalid(obs, minval=minval), util._idvalid(est, minval=minval)
        )
        self.n = len(self.ix)
        if self.n == 0:
            warnings.warn(
                "WRADLIB: No valid pairs of observed and "
                "estimated available for ErrorMetrics!"
            )
        self.resids = self.est[self.ix] - self.obs[self.ix]

    def corr(self):
        """Correlation coefficient
        """
        return np.round(np.corrcoef(self.obs[self.ix], self.est[self.ix])[0, 1], 2)

    def r2(self):
        """Coefficient of determination
        """
        return np.round(
            (np.corrcoef(self.obs[self.ix], self.est[self.ix])[0, 1]) ** 2, 2
        )

    def spearman(self):
        """Spearman rank correlation coefficient
        """
        return np.round(
            stats.stats.spearmanr(self.obs[self.ix], self.est[self.ix])[0], 2
        )

    def nash(self):
        """Nash-Sutcliffe Efficiency
        """
        return np.round(1.0 - (self.mse() / np.var(self.obs[self.ix])), 2)

    def sse(self):
        """Sum of Squared Errors
        """
        return np.round(np.sum(self.resids ** 2), 2)

    def mse(self):
        """Mean Squared Error
        """
        return np.round(self.sse() / self.n, 2)

    def rmse(self):
        """Root Mean Squared Error
        """
        return np.round(self.mse() ** 0.5, 2)

    def mas(self):
        """Mean Absolute Error
        """
        return np.round(np.mean(np.abs(self.resids)), 2)

    def meanerr(self):
        """Mean Error
        """
        return np.round(np.mean(self.resids), 2)

    def ratio(self):
        """Mean ratio between observed and estimated
        """
        return np.round(np.mean(self.est[self.ix] / self.obs[self.ix]), 2)

    def pbias(self):
        """Percent bias
        """
        return np.round(self.meanerr() * 100.0 / np.mean(self.obs[self.ix]), 1)

    def all(self):
        """Returns a dictionary of all error metrics
        """
        out = {
            "corr": self.corr(),
            "r2": self.r2(),
            "spearman": self.spearman(),
            "nash": self.nash(),
            "sse": self.sse(),
            "mse": self.mse(),
            "rmse": self.rmse(),
            "mas": self.mas(),
            "meanerr": self.meanerr(),
            "ratio": self.ratio(),
            "pbias": self.pbias(),
        }

        return out

    def pprint(self):
        """Pretty prints a summary of error metrics
        """
        pprint(self.all())


if __name__ == "__main__":
    print("wradlib: Calling module <verify> as main...")
