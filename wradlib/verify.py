#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Verification
^^^^^^^^^^^^

Verification mainly refers to the comparison of radar-based precipitation
estimates to ground truth.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ErrorMetrics
   PolarNeighbours

"""
# site packages
import numpy as np
from scipy.spatial import KDTree
from scipy import stats
import matplotlib.pyplot as pl
from pprint import pprint

# wradlib modules
from . import georef as georef
from . import util as util


class PolarNeighbours():
    """
    For a set of projected point coordinates, extract the neighbouring bin
    values from a data set in polar coordinates. Use as follows:

    First, create an instance of PolarNeighbours by passing all the information
    needed to georeference the polar radar data to the points of interest
    (see parameters).

    Second, use the method *extract* in order to extract the values from a data
    array which corresponds to the polar coordinates.

    .. versionchanged:: 0.5.0
       using osr objects instead of PROJ.4 strings as parameter

    Parameters
    ----------
    r : array of floats
        (see georef for documentation)
    az : array of floats
        (see georef for documentation)
    sitecoords : sequence of floats
        (see georef for documentation)
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection
        (see georef for documentation)
    x : array of floats
        x coordinates of the points in map projection corresponding to proj
    y : array of floats
        y coordinates of the points in map projection corresponding to proj
    nnear : int
        number of neighbouring radar bins you would like to find

    Examples
    --------

    See :ref:`notebooks/verification/wradlib_verify_example.ipynb`.

    """

    def __init__(self, r, az, sitecoords, proj, x, y, nnear=9):
        self.nnear = nnear
        self.az = az
        self.r = r
        self.x = x
        self.y = y
        # compute the centroid coordinates in lat/lon
        bin_lon, bin_lat = georef.polar2centroids(r, az, sitecoords)
        # reproject the centroids to cartesian map coordinates
        binx, biny = georef.reproject(bin_lon, bin_lat,
                                      projection_target=proj)
        self.binx, self.biny = binx.ravel(), biny.ravel()
        # compute the KDTree
        tree = KDTree(list(zip(self.binx, self.biny)))
        # query the tree for nearest neighbours
        self.dist, self.ix = tree.query(list(zip(x, y)), k=nnear)

    def extract(self, vals):
        """
        Extracts the values from an array of shape (azimuth angles,
        range gages) which correspond to the indices computed during
        initialisation

        Parameters
        ----------
        vals : array of shape (..., number of azimuth, number of range gates)

        Returns
        -------
        output : array of shape (..., number of points, nnear)

        """
        assert vals.ndim >= 2, \
            'Your <vals> array should at least contain an ' \
            'azimuth and a range dimension.'
        assert tuple(vals.shape[-2:]) == (len(self.az), len(self.r)), \
            'The shape of your vals array does not correspond with ' \
            'the range and azimuths you provided for your polar data set'
        vals = vals.reshape(vals.shape[:-2] + (len(self.az) * len(self.r),))
        return vals[..., self.ix]

    def get_bincoords(self):
        """
        Returns all bin coordinates in map projection

        Returns
        -------
        output : array of x coordinates, array of y coordinates

        """
        return self.binx, self.biny

    def get_bincoords_at_points(self):
        """
        Returns bin coordinates only in the neighbourhood of points

        Returns
        -------
        output : array of x coordinates, array of y coordinates

        """
        return self.binx[self.ix], self.biny[self.ix]


class ErrorMetrics():
    """Compute quality metrics from a set of observations (obs) and
    estimates (est).

    First create an instance of the class using the set of observations and
    estimates. Then compute quality metrics using the class methods.
    A dictionary of all available quality metrics is returned using the
    *all* method. Method *report* pretty prints all these metrics over a
    scatter plot.

    Parameters
    ----------
    obs: array of floats
        observations (e.g. rain gage observations)
    est: array of floats
        estimates (e.g. radar, adjusted radar, ...)
    minval : float
        threshold value in order to compute metrics only for values larger
        than minval

    Examples
    --------
    >>> obs = np.random.uniform(0, 10, 100)
    >>> est = np.random.uniform(0, 10, 100)
    >>> metrics = ErrorMetrics(obs, est)
    >>> metrics.all() # doctest: +SKIP
    >>> metrics.pprint() # doctest: +SKIP
    >>> ax = metrics.plot() # doctest: +SKIP
    >>> metrics.report() # doctest: +SKIP

    See :ref:`notebooks/verification/wradlib_verify_example.ipynb` and
    :ref:`notebooks/multisensor/wradlib_adjust_example.ipynb`.

    """

    def __init__(self, obs, est, minval=None):
        # Check input
        assert len(obs) == len(est), \
            "obs and est need to have the same length. " \
            "len(obs)=%d, len(est)=%d" % (len(obs), len(est))
        # only remember those entries which have both valid observations
        # AND estimates
        ix = np.intersect1d(util._idvalid(obs, minval=minval),
                            util._idvalid(est, minval=minval))
        self.n = len(ix)
        if self.n == 0:
            print("WARNING: No valid pairs of observed and "
                  "estimated available for ErrorMetrics!")
            self.obs = np.array([])
            self.est = np.array([])
        else:
            self.obs = obs[ix]
            self.est = est[ix]
        self.resids = self.est - self.obs

    def corr(self):
        """Correlation coefficient
        """
        return np.round(np.corrcoef(self.obs, self.est)[0, 1], 2)

    def r2(self):
        """Coefficient of determination
        """
        return np.round((np.corrcoef(self.obs, self.est)[0, 1]) ** 2, 2)

    def spearman(self):
        """Spearman rank correlation coefficient
        """
        return np.round(stats.stats.spearmanr(self.obs, self.est)[0], 2)

    def nash(self):
        """Nash-Sutcliffe Efficiency
        """
        return np.round(1. - (self.mse() / np.var(self.obs)), 2)

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
        return np.round(np.mean(self.est / self.obs), 2)

    def pbias(self):
        """Percent bias
        """
        return np.round(self.meanerr() * 100. / np.mean(self.obs), 1)

    def all(self):
        """Returns a dictionary of all error metrics
        """
        out = {"corr": self.corr(),
               "r2": self.r2(),
               "spearman": self.spearman(),
               "nash": self.nash(),
               "sse": self.sse(),
               "mse": self.mse(),
               "rmse": self.rmse(),
               "mas": self.mas(),
               "meanerr": self.meanerr(),
               "ratio": self.ratio(),
               "pbias": self.pbias()}

        return out

    def plot(self, ax=None, unit="", maxval=None):
        """Scatter plot of estimates vs observations

        Parameters
        ----------
        ax : a matplotlib axes object to plot on
           if None, a new axes object will be created
        unit : string
           measurement unit of the observations / estimates
        maxval : maximum value for plot range, defaults to max(obs, est)
        """
        if self.n == 0:
            print("No valid data, no plot.")
            return None
        doplot = False
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, aspect=1.)
            doplot = True
        ax.plot(self.obs, self.est, mfc="None", mec="black", marker="o", lw=0)
        if maxval is None:
            maxval = np.max(np.append(self.obs, self.est))
        pl.xlim(xmin=0., xmax=maxval)
        pl.ylim(ymin=0., ymax=maxval)
        ax.plot([0, maxval], [0, maxval], "-", color="grey")
        pl.xlabel("Observations (%s)" % unit)
        pl.ylabel("Estimates (%s)" % unit)
        if (not pl.isinteractive()) and doplot:
            pl.show()
        return ax

    def pprint(self):
        """Pretty prints a summary of error metrics
        """
        pprint(self.all())

    def report(self, metrics=None, ax=None, unit="", maxval=None):
        """Pretty prints selected error metrics over a scatter plot

        Parameters
        ----------
        metrics : sequence of strings
           names of the metrics which should be included in the report
           defaults to ["rmse","r2","meanerr"]
        ax : a matplotlib axes object to plot on
           if None, a new axes object will be created
        unit : string
           measurement unit of the observations / estimates

        """
        if self.n == 0:
            print("No valid data, no report.")
            return None
        if metrics is None:
            metrics = ["rmse", "nash", "pbias"]
        doplot = False
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, aspect=1.)
            doplot = True
        ax = self.plot(ax=ax, unit=unit, maxval=maxval)
        if maxval is None:
            maxval = np.max(np.append(self.obs, self.est))
        xtext = 0.6 * maxval
        ytext = (0.1 + np.arange(0, len(metrics), 0.1)) * maxval
        mymetrics = self.all()
        for i, metric in enumerate(metrics):
            pl.text(xtext, ytext[i], "%s: %s" % (metric, mymetrics[metric]))
        if not pl.isinteractive() and doplot:
            pl.show()


if __name__ == '__main__':
    print('wradlib: Calling module <verify> as main...')
