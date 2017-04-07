#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Interpolation
^^^^^^^^^^^^^

Interpolation allows to transfer data from one set of locations to another.
This includes for example:

- interpolating the data from a polar grid to a cartesian grid or irregular
  points

- interpolating point observations to a grid or a set of irregular points

- filling missing values, e.g. filling clutters

.. autosummary::
   :nosignatures:
   :toctree: generated/

   Nearest
   Idw
   Linear
   OrdinaryKriging
   ExternalDriftKriging
   interpolate
   interpolate_polar
   cart2irregular_interp
   cart2irregular_spline

"""

from functools import reduce
import re
import scipy
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import griddata
import numpy as np
import warnings

from . import util as util


class MissingSourcesError(Exception):
    """Is raised in case no source coordinates are available for interpolation.
    """
    pass


class MissingTargetsError(Exception):
    """Is raised in case no interpolation targets are available.
    """
    pass


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
        self.numsources = len(src)
        self.numtargets = len(trg)

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
        assert len(vals) == self.numsources, \
            ('Length of value array %d does not correspond to number '
             'of source points %d' % (len(vals), self.numsources))

    def _make_coord_arrays(self, x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)

        Parameters
        ----------
        x : ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and
            the length of the ndarray corresponding to the number of points

        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif type(x) == np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 2:
                pass
            else:
                raise Exception('Cannot deal wih 3-d arrays, yet.')
        return x

    def _make_2d(self, vals):
        """Reshape increase number of dimensions of vals if smaller than 2,
        appending additional dimensions (as opposed to the atleast_nd methods
        of numpy).

        Parameters
        ----------
        vals : ndarray
               values who are to be reshaped to the right shape

        Returns
        -------
        output : ndarray
                 if vals.shape==() [a scalar] output.shape will be (1,1)
                 if vals.shape==(npt,) output.shape will be (npt,1)
                 if vals.ndim > 1 vals will be returned as is
        """
        if vals.ndim < 2:
            # ndmin might be 0 so we get it to 1-d first
            # then we add an axis as we assume that
            return np.atleast_1d(vals)[:, np.newaxis]
        else:
            return vals


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

    Examples
    --------
    See :ref:`notebooks/interpolation/wradlib_ipol_example.ipynb`.


    Note
    ----
    Uses :class:`scipy:scipy.spatial.cKDTree`

    """

    def __init__(self, src, trg):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError
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
        if maxdist is None:
            return out
        else:
            return np.where(self.dists > maxdist, np.nan, out)


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

    Examples
    --------
    See :ref:`notebooks/interpolation/wradlib_ipol_example.ipynb`.

    Note
    ----
    Uses :class:`scipy:scipy.spatial.cKDTree`

    """

    def __init__(self, src, trg, nnearest=4, p=2.):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError
        if nnearest > self.numsources:
            warnings.warn(
                "wradlib.ipol.Idw: <nnearest> is larger than number of "
                "source points and is set to %d corresponding to the "
                "number of source points." % self.numsources,
                UserWarning
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest
        self.p = p
        # plant a tree
        self.tree = cKDTree(src)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:, np.newaxis]
            self.ix = self.ix[:, np.newaxis]

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

        # self distances: a list of arrays of distances of the nearest points
        # which are indicated by self.ix
        self._check_shape(vals)
        outshape = list(vals.shape)
        outshape[0] = len(self.dists)
        interpol = (np.repeat(np.nan, util._shape2size(outshape)).
                    reshape(tuple(outshape)).astype('f4'))
        # weights is the container for the weights (a list)
        weights = list(range(len(self.dists)))
        # sources is the container for the source point indices
        src_ix = list(range(len(self.dists)))
        # jinterpol is the jth element of interpol
        jinterpol = 0
        for dist, ix in zip(self.dists, self.ix):
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
                w = 1.
            else:
                # weight z values by (1/dist)**p --
                w = 1. / dist ** self.p
                w /= np.sum(w)
                wz = np.dot(w, vals[ix])
            interpol[jinterpol] = wz.ravel()
            weights[jinterpol] = w
            src_ix[jinterpol] = ix
            jinterpol += 1
        return interpol  # if self.qdim > 1  else interpol[0]


class Linear(IpolBase):
    """
    Interface to the :class:`scipy:scipy.interpolate.LinearNDInterpolator`
    class.

    We provide this class in order to achieve a uniform interface for all
    Interpolator classes

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.

    Examples
    --------
    See :ref:`notebooks/interpolation/wradlib_ipol_example.ipynb`.
    """

    def __init__(self, src, trg):
        self.src = self._make_coord_arrays(src)
        self.trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError

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


# -----------------------------------------------------------------------------
# Covariance routines needed for Kriging
# -----------------------------------------------------------------------------
def parse_covariogram(cov_model):
    """"""
    patterns = [re.compile('([\d\.]+) Nug\(([\d\.]+)\)'),  # nugget
                re.compile('([\d\.]+) Lin\(([\d\.]+)\)'),  # linear
                re.compile('([\d\.]+) Sph\(([\d\.]+)\)'),  # spherical
                re.compile('([\d\.]+) Exp\(([\d\.]+)\)'),  # exponential
                re.compile('([\d\.]+) Gau\(([\d\.]+)\)'),  # gaussian
                re.compile('([\d\.]+) Mat\(([\d\.]+)\)\^([\d\.]+)'),  # matern
                re.compile('([\d\.]+) Pow\(([\d\.]+)\)'),  # power
                # cauchy
                re.compile('([\d\.]+) Cau\(([\d\.]+)\)\^([\d\.]+)\^([\d\.]+)'),
                ]

    cov_funs = [cov_nug,
                cov_lin,
                cov_sph,
                cov_exp,
                cov_gau,
                cov_mat,
                cov_pow,
                cov_cau,
                ]

    funcs = []

    # first split along '+'
    subparts = cov_model.split('+')
    # then analyse subparts
    for i, subpart in enumerate(subparts):
        # iterate over all available patterns
        for j, pattern in enumerate(patterns):
            m = pattern.search(subpart)
            if m:
                params = [float(p) for p in m.groups()]
                funcs.append(_make_cov(cov_funs[j], params))

    # return complete covariance function, which adds
    # individual subparts
    return lambda h: reduce(np.add, [f(h) for f in funcs])


def _make_cov(func, params):
    return lambda h: func(h, *params)


def cov_nug(h, sill, rng):
    r"""nugget covariance function
    :math:`\gamma(h) = s ` for :math:`h \leq r`, 0 otherwise
    Therefore, usually rng is set to 0
    """
    h = np.asanyarray(h)
    return np.where(h <= rng, sill, 0.)


def cov_exp(h, sill=1.0, rng=1.0):
    """exponential type covariance function"""
    h = np.asanyarray(h)
    return sill * (np.exp(-h / rng))


def cov_sph(h, sill=1.0, rng=1.0):
    """spherical type covariance function"""
    h = np.asanyarray(h)
    return np.where(h < rng, sill * (1. - 1.5 * h /
                                     rng + h ** 3 /
                                     (2 * rng ** 3)), 0.)


def cov_gau(h, sill=1.0, rng=1.0):
    """gaussian type covariance function"""
    h = np.asanyarray(h)
    return sill * np.exp(-h ** 2 / rng ** 2)


def cov_lin(h, sill=1.0, rng=1.0):
    """linear covariance function"""
    h = np.asanyarray(h)
    return np.where(h < rng, sill * (-h / rng + 1.), 0.)


def cov_mat(h, sill=1.0, rng=1.0, shp=0.5):
    """matern covariance function"""
    """Matern Covariance Function Family:
        shp = 0.5 --> Exponential Model
        shp = inf --> Gaussian Model
    """
    h = np.asanyarray(h)

    # for v > 100 shit happens --> use Gaussian model
    if shp > 100:
        c = cov_gau(h, sill, rng)
    else:
        # modified bessel function of second kind of order v
        Kv = scipy.special.kv
        # Gamma function
        Tau = scipy.special.gamma

        fac1 = h / rng * 2.0 * np.sqrt(shp)
        fac2 = (Tau(shp) * 2.0 ** (shp - 1.0))

        c = np.where(h != 0, sill * 1.0 /
                     fac2 * fac1 ** shp * Kv(shp, fac1), sill)

    return c


def cov_pow(h, sill=1.0, rng=1.0):
    """power law covariance function"""
    h = np.asanyarray(h)
    return sill - h ** rng


def cov_cau(h, sill=1., rng=1., alpha=1., beta=1.0):
    """
    cauchy covariance function.

    alpha >0 & <=2 ... shape parameter
    beta >0 ... parameterizes long term memory
    """
    h = np.asanyarray(h).astype('float')
    return sill * (1 + (h / rng) ** alpha) ** (-beta / alpha)


class OrdinaryKriging(IpolBase):
    r"""
    OrdinaryKriging(src, trg, cov='1.0 Exp(10000.)', nnearest=12)

    Interpolate using Ordinary Kriging

    (Co-)Variogram definitions are given in the syntax that ``gstat`` uses.
    It allows nesting of different basic variogram types using linear
    combinations.
    Each basic covariogram is usually defined by
    Note that, strictly speaking, this implementation doesn't allow Kriging of
    fields for which the covariance does not exist. While this is
    mathematically possible, it is rather rare for fields encountered in
    reality. Therefore, this should not be a severe limitation.

    Most (co-)variograms are characterized by a sill parameter (which is
    the (co-)variance at separation distance 0) a range parameter (which
    indicates a separation distance after which the the covariance drops
    close to zero) an sometimes additional parameters governing the shape
    of the function. In the following range is given by the variable `r` and
    the sill by the variable `s`.
    Currently implemented are:

        - Pure Nugget
        - Exponential
        - Spherical
        - Gaussian
        - Linear
        - Matern
        - Power
        - Cauchy

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    cov : string
        covariance (variogram) model string in the syntax ``gstat``
        uses.
    nnearest : integer - max. number of neighbours to be considered

    Note
    ----
    The class calculates the Kriging weights during initialization, because
    these only depend on the configuration of the points.

    The call method is then only used to calculate estimated values at the
    target points based on those at the source points. Therefore the main
    computational load is experienced during initialization. This behavior is
    different from that of the Idw or Nearest Interpolators.

    After initialization the estimation variance at each interpolation target
    may be retrieved from the attribute `estimation_variance`.

    Examples
    --------
    See :ref:`notebooks/interpolation/wradlib_ipol_example.ipynb`.
    """

    def __init__(self, src, trg, cov='1.0 Exp(10000.)', nnearest=12):
        """"""
        self.src = self._make_coord_arrays(src)
        self.trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError
        if nnearest > self.numsources:
            warnings.warn(
                "wradlib.ipol.OrdinaryKriging: <nnearest> is "
                "larger than number of source points and is "
                "set to %d corresponding to the "
                "number of source points." % self.numsources,
                UserWarning
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest
        # plant a tree
        self.tree = cKDTree(src)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:, np.newaxis]
            self.ix = self.ix[:, np.newaxis]
        # parse covariogram function string
        self.cov_func = parse_covariogram(cov)
        self.weights = []
        self.estimation_variance = []
        # do the kriging
        self._krige()

    def _krig_matrix(self, src):
        """Sets up the kriging system for a configuration of source points.
        """
        var_matrix = self.cov_func(scipy.spatial.distance_matrix(src, src))

        ok_matrix = np.ones((len(src) + 1, len(src) + 1))

        ok_matrix[:-1, :-1] = var_matrix
        ok_matrix[-1, -1] = 0.

        return ok_matrix

    def _krig_rhs(self, dists):
        """Sets up a right hand side of the kriging system given the distances
        of the target to the source points. To be used in conjunction with
        `_krig_matrix`."""
        rhs = self.cov_func(dists)
        ok_rhs = np.concatenate([rhs, [1.]])

        return ok_rhs

    def _krige(self):
        """Sets up the kriging system and solves it in order to obtain the
        interpolation weights of ordinary kriging.
        Also calculates the kriging estimation variance from the results"""
        for dist, ix in zip(self.dists, self.ix):
            matrix = self._krig_matrix(self.src[ix, :])
            rhs = self._krig_rhs(dist)
            weights = np.linalg.solve(matrix, rhs)
            self.weights.append(weights)
            self.estimation_variance.append(self.cov_func(0.) -
                                            np.sum(weights * rhs))

    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsourcepoints, numfields)
            Values at the source points from which to interpolate
            Several fields may be calculated at once by passing them
            along the second dimension.
            Only this second dimension is implemented. You'll have to
            reshape a more complex array for the function to work.

        Returns
        -------
        output : ndarray of float with shape (numtargetpoints, numfields)

        """
        v = self._make_2d(vals)
        self._check_shape(v)
        # calculate estimator
        weights = np.array(self.weights)
        ip = np.add.reduce(weights[:, :-1, np.newaxis] * v[self.ix, ...],
                           axis=1)

        return ip


class ExternalDriftKriging(IpolBase):
    """
    ExternalDriftKriging(src, trg, cov='1.0 Exp(10000.)', nnearest=12,
                         drift_src=None, drift_trg=None)

    Kriging with external drift

    Parameters
    ----------
    src : ndarray of floats, shape (nsrcpoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (ntrgpoints, ndims)
        Data point coordinates of the target points.
    cov : string
        covariance (variogram) model string in the syntax ``gstat``
        uses.
    nnearest : int
        max. number of neighbours to be considered
    src_drift : ndarray of floats, shape (nsrcpoints,)
        values of the external drift at each source point
    trg_drift : ndarray of floats, shape (ntrgpoints,)
        values of the external drift at each target point

    See Also
    --------
    OrdinaryKriging

    Note
    ----
    After calling the object in order to get the interpolated values,
    the estimation variance of the system may be
    retrieved from the attribute `estimation_variance`. Accordingly, the
    interpolation weights can be retrieved from the attribute `weights`

    If drift_src or drift_trg are not given on initialization, they must
    be provided when using the __call__ method.
    If any of them is given on initialization its values may be overridden
    by passing new values to the __call__ method.

    """

    def __init__(self, src, trg, cov='1.0 Exp(10000.)', nnearest=12,
                 src_drift=None, trg_drift=None):
        """"""
        self.src = self._make_coord_arrays(src)
        self.trg = self._make_coord_arrays(trg)
        self.src_drift = src_drift
        self.trg_drift = trg_drift
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError
        if nnearest > self.numsources:
            warnings.warn(
                "wradlib.ipol.ExternalDriftKriging: <nnearest> is larger "
                "than number of source points and is set to %d "
                "corresponding to the number of source "
                "points." % self.numsources,
                UserWarning
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest
        # plant a tree
        self.tree = cKDTree(src)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:, np.newaxis]
            self.ix = self.ix[:, np.newaxis]
        # parse covariogram function string
        self.cov_func = parse_covariogram(cov)
        self.weights = []
        self.estimation_variance = []

    def _krig_matrix(self, src, drift):
        """Sets up the kriging system for a configuration of source points.
        """
        # the basic covariance matrix
        var_matrix = self.cov_func(scipy.spatial.distance_matrix(src, src))
        # the extended matrix, initialized to ones
        edk_matrix = np.ones((len(src) + 2, len(src) + 2))

        # adding entries for the first lagrange multiplier for the ordinary
        # kriging part
        edk_matrix[:-2, :-2] = var_matrix
        edk_matrix[-2, -2] = 0.

        # adding entries for the second lagrange multiplier for the  edk part
        edk_matrix[:-2, -1] = drift
        edk_matrix[-1, :-2] = drift
        edk_matrix[-2:, -1] = 0.
        edk_matrix[-1, -2:] = 0.

        return edk_matrix

    def _krig_rhs(self, dists, drift):
        """Sets up a right hand side of the kriging system given the distances
        of the target to the source points. To be used in conjunction with
        `_krig_matrix`."""
        rhs = self.cov_func(dists)
        edk_rhs = np.concatenate([rhs, np.array([1., drift])])

        return edk_rhs

    def _krige(self, src_drift, trg_drift):
        """Sets up the kriging system and solves it in order to obtain the
        interpolation weights of ordinary kriging.
        Also calculates the kriging estimation variance from the results"""
        all_weights = []
        estimation_variances = []
        for dist, ix, td in zip(self.dists, self.ix, trg_drift):
            matrix = self._krig_matrix(self.src[ix, :], src_drift[ix])
            rhs = self._krig_rhs(dist, td)
            try:
                weights = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                weights = np.repeat(np.nan, len(rhs))
            all_weights.append(weights)
            estimation_variances.append(self.cov_func(0.) -
                                        np.sum(weights * rhs))

        return all_weights, estimation_variances

    def __call__(self, vals, src_drift=None, trg_drift=None):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : ndarray of float, shape (numsourcepoints, numfields)
            Values at the source points from which to interpolate
            Several fields may be calculated at once by passing them
            along the second dimension.
            Only this second dimension is implemented. You'll have to
            reshape a more complex array for the function to work.

        Returns
        -------
        output : ndarray of float with shape (numtargetpoints, numfields)

        """
        assert vals.ndim <= 2
        v = self._make_2d(vals)
        self._check_shape(v)

        if src_drift is None:
            # check if we have data from __init__
            if self.src_drift is None:
                raise ValueError('src_drift must be specified either on '
                                 'initialization or when calling '
                                 'the interpolator.')
            src_drift = self.src_drift
        if trg_drift is None:
            # check if we have data from __init__
            if self.trg_drift is None:
                raise ValueError('trg_drift must be specified either on '
                                 'initialization or when calling the '
                                 'interpolator.')
            trg_drift = self.trg_drift

        src_d = self._make_2d(src_drift)
        trg_d = self._make_2d(trg_drift)
        self._check_shape(src_d)

        # re-initialize weights and variances to ensure that these only reflect
        # the results of the current call and not any previous call
        self.weights = []
        self.estimation_variance = []

        # if drifts are constant, we can save time by solving the kriging
        # system once
        if src_d.shape[1] == 1:
            wght, variances = self._krige(src_d.squeeze(), trg_d.squeeze())
            self.weights = wght
            self.estimation_variance = variances
            weights = np.array(self.weights)
            ip = np.add.reduce(weights[:, :-2, np.newaxis] * v[self.ix, ...],
                               axis=1)
        # otherwise we need to setup and solve the kriging system for each
        # field individually
        else:
            ip = np.empty((self.trg.shape[0], v.shape[1]))
            assert ((v.shape[1] == src_d.shape[1]) and
                    (v.shape[1] == trg_d.shape[1]))
            for i in range(v.shape[1]):
                wght, variances = self._krige(src_d[:, i].squeeze(),
                                              trg_d[:, i].squeeze())

                weights = np.array(wght)
                ip[:, i] = np.add.reduce(weights[:, :-2] * v[self.ix, i],
                                         axis=1)
                self.weights.append(weights)
                self.estimation_variance.append(variances)

        return ip


# -----------------------------------------------------------------------------
# Wrapper functions
# -----------------------------------------------------------------------------
def interpolate(src, trg, vals, Interpolator, *args, **kwargs):
    """
    Convenience function to use the interpolation classes in an efficient way

    The interpolation classes in wradlib.ipol are computationally very
    efficient if they are applied on large multi-dimensional arrays of which
    the first dimension must be the locations' dimension (1d or 2d coordinates)
    and the following dimensions can be anything (e.g. time or ensembles). This
    way, the weights need to be computed only once. However, this can only be
    done with success if all source values for the interpolation are valid
    numbers. If the source values contain let's say *np.nan* types, the result
    of the interpolation will be *np.nan* in the vicinity of the corresponding
    points, too. Imagine that you have a time series of observations at points
    and in each time step one observation is missing.
    You would still like to efficiently apply the interpolation
    classes, but you will need to account for the resulting *np.nan* values in
    the interpolation output.

    In order to still allow for the efficient application, you have to take
    care of the remaining np.nan in your interpolation result. This is done by
    this convenience function.

    Alternatively, you have to make sure that your *vals* argument does not
    contain any *np.nan* values OR you have to post-process missing values in
    your interpolation result in another way.

    Warning
    -------
    Works only for one- and two-dimensional *vals* arrays, yet.

    Parameters
    ----------
    src : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    vals : ndarray of float, shape (numsourcepoints, ...)
        Values at the source points which to interpolate
    Interpolator : a class which inherits from IpolBase

    Other Parameters
    ----------------
    *args : arguments of Interpolator (see class documentation)

    Keyword Arguments
    -----------------
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
    >>> import matplotlib.pyplot as plt
    >>> plt.interactive(True)
    >>> line1 = plt.plot(trg, ipol_result, 'b+')
    >>> line2 = plt.plot(src, vals, 'ro')

    """
    if vals.ndim == 1:
        # source values are one dimensional, we have just
        # to remove invalid data
        ix_valid = np.where(np.isfinite(vals))[0]
        ip = Interpolator(src[ix_valid], trg, *args, **kwargs)
        result = ip(vals[ix_valid])
    elif vals.ndim == 2:
        # this implementation for 2 dimensions needs generalization
        ip = Interpolator(src, trg, *args, **kwargs)
        result = ip(vals)
        nan_in_result = np.where(np.isnan(result))
        # nan_in_vals = np.where(np.isnan(vals))
        for i in np.unique(nan_in_result[-1]):
            ix_good = np.where(np.isfinite(vals[..., i]))[0]
            ix_broken_targets = (nan_in_result[0]
                                 [np.where(nan_in_result[-1] == i)[0]])
            ip = Interpolator(src[ix_good],
                              trg[nan_in_result[0]
                              [np.where(nan_in_result[-1] == i)[0]]],
                              *args, **kwargs)
            tmp = ip(vals[ix_good, i].reshape((len(ix_good), -1)))
            result[ix_broken_targets, i] = tmp.ravel()
    else:
        if not np.any(np.isnan(vals.ravel())):
            raise Exception(
                'At the moment, <interpolate> can only deal with NaN values '
                'in <vals> if vals has less than 3 dimension.')
        else:
            # if no NaN value are in <vals> we can safely apply the
            # Interpolator as is
            ip = Interpolator(src, trg, *args, **kwargs)
            result = ip(vals[ix_valid])
    return result


def interpolate_polar(data, mask=None, Interpolator=Nearest):
    """
    Convenience function to interpolate polar data

    Parameters
    ----------
    data : 2d-array
        2 dimensional array (azimuth, ranges) of floats;

        if no mask is assigned explicitly polar data should be a masked array
    mask : array
        boolean array with pixels to be interpolated set to True;

        must have the same shape as data
    Interpolator : a class which inherits from IpolBase

    Returns
    -------
    filled_data : 2d-array
        array with interpolated values for the values set to True in the mask

    Examples
    --------
    >>> import numpy as np  # noqa
    >>> import wradlib as wrl
    >>> # creating a data array and mask some values
    >>> data = np.arange(12.).reshape(4,3)
    >>> masked_values = (data==2) | (data==9)
    >>> # interpolate the masked data based on ''masked_values''
    >>> filled_a = wrl.ipol.interpolate_polar(data, mask = masked_values, Interpolator = wrl.ipol.Linear)  # noqa
    >>> ax, pm = wrl.vis.plot_ppi(filled_a)
    >>> # the same result can be achieved by using an masked array instead of an explicit mask  # noqa
    >>> mdata = np.ma.array(data, mask = masked_values)
    >>> filled_b = wrl.ipol.interpolate_polar(mdata, Interpolator = wrl.ipol.Linear)  # noqa
    >>> ax, pm = wrl.vis.plot_ppi(filled_b)


    """
    if mask is None:
        # no mask assigned: try to get it from masked array
        if type(data) != np.ma.core.MaskedArray:
            print('Warning! Neither an explicit mask is assigned nor the '
                  'data-array is masked.')
        mask = np.ma.getmaskarray(data)
    elif not np.any(mask):
        # mask contains no True values, so there is nothing to fill
        return data
    clutter_indices = np.where(mask.ravel())
    # construct the ranges for every bin
    ranges = np.tile(np.arange(0.5, data.shape[1] + 0.5), data.shape[0])
    # construct the angles for every bin
    angles = np.repeat(np.radians(np.arange(0, 360, 360. / data.shape[0])),
                       data.shape[1])
    # calculate cartesian coordinates for every bin
    binx = np.cos(angles) * ranges
    biny = np.sin(angles) * ranges
    # calculate cartesian coordinates for bins, which are not masked
    src_coord = np.array([(np.delete(binx, clutter_indices)),
                          (np.delete(biny, clutter_indices))]).transpose()
    # calculate cartesian coordinates for bins, which are masked
    trg_coord = np.array([binx[clutter_indices],
                          biny[clutter_indices]]).transpose()
    # data values for bins, which are not masked
    values_list = np.delete(data, clutter_indices)
    filled_data = data.copy().ravel()
    # interpolate masked bins
    filling = interpolate(src_coord, trg_coord, values_list, Interpolator)
    # fill data with the interpolations
    filled_data[clutter_indices] = filling.astype(filled_data.dtype)
    # in case of nans as processed at the rim when interpolating linear,
    # these values are finally interpolated by nearest Neighbor interpolation
    if np.any(np.isnan(filled_data)):
        trg_coord = (np.array([binx[np.where(np.isnan(filled_data))],
                              biny[np.where(np.isnan(filled_data))]])
                     .transpose())
        filling = interpolate(src_coord, trg_coord, values_list,
                              Interpolator=Nearest)
        filled_data[np.where(np.isnan(filled_data))] = filling
    return filled_data.reshape(data.shape[0], data.shape[1])


def cart2irregular_interp(cartgrid, values, newgrid, **kwargs):
    """
    Interpolate array ``values`` defined by cartesian coordinate array
    ``cartgrid`` to new coordinates defined by ``newgrid`` using
    nearest neighbour, linear or cubic interpolation

    .. versionadded:: 0.6.0

    Slow for large arrays

    Keyword arguments are fed to :func:`scipy:scipy.interpolate.griddata`

    Parameters
    ----------
    cartgrid : numpy ndarray
        3 dimensional array (nx, ny, lon/lat) of floats;
    values : numpy 2d-array
        2 dimensional array (nx, ny) of data values
    newgrid : numpy ndarray
        Nx2 dimensional array (..., lon/lat) of floats
    kwargs : :func:`scipy:scipy.interpolate.griddata`

    Returns
    -------
    interp : numpy ndarray
        array with interpolated values of size N
    """

    # TODO: dimension checking

    newshape = newgrid.shape[:-1]

    cart_arr = cartgrid.reshape(-1, cartgrid.shape[-1])
    new_arr = newgrid.reshape(-1, newgrid.shape[-1])

    if values.ndim > 1:
        values = values.ravel()

    interp = griddata(cart_arr, values, new_arr, **kwargs)
    interp = interp.reshape(newshape)

    return interp


def cart2irregular_spline(cartgrid, values, newgrid, **kwargs):
    """
    Map array ``values`` defined by cartesian coordinate array ``cartgrid``
    to new coordinates defined by ``newgrid`` using spline interpolation.

    .. versionadded:: 0.6.0

    .. versionchanged:: 0.10.0
       Accept data/coords with origin 'lower' or 'upper'.

    Keyword arguments are fed through to
    :func:`scipy:scipy.ndimage.map_coordinates`

    Parameters
    ----------
    cartgrid : numpy ndarray
        3 dimensional array (nx, ny, lon/lat) of floats
    values : numpy 2d-array
        2 dimensional array (nx, ny) of data values
    newgrid : numpy ndarray
        Nx2 dimensional array (..., lon/lat) of floats
    kwargs : :func:`scipy:scipy.ndimage.map_coordinates`

    Returns
    -------
    interp : numpy ndarray
        array with interpolated values of size N

    Examples
    --------
    See :ref:`notebooks/beamblockage/wradlib_beamblock.ipynb#\
Read-DEM-Raster-Data`.
    """

    # TODO: dimension checking
    newshape = newgrid.shape[:-1]

    xi = newgrid[..., 0].ravel()
    yi = newgrid[..., 1].ravel()

    nx = cartgrid.shape[1]
    ny = cartgrid.shape[0]

    cxmin = np.min(cartgrid[..., 0])
    cxmax = np.max(cartgrid[..., 0])
    cymin = np.min(cartgrid[..., 1])
    cymax = np.max(cartgrid[..., 1])

    # this functionality finds the floating point
    # indices into the value array (0:nx-1)
    # can be transferred into separate function
    # if necessary
    xi = (nx - 1) * (xi - cxmin) / (cxmax - cxmin)

    # check origin to calculate y index
    if util.get_raster_origin(cartgrid) is 'lower':
        yi = (ny - 1) * (yi - cymin) / (cymax - cymin)
    else:
        yi = ny - (ny - 1) * (yi - cymin) / (cymax - cymin)

    # interpolation by map_coordinates
    interp = map_coordinates(values, [yi, xi], **kwargs)
    interp = interp.reshape(newshape)

    return interp


if __name__ == '__main__':
    print('wradlib: Calling module <ipol> as main...')
