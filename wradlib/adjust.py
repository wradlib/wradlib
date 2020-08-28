#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Gage adjustment
^^^^^^^^^^^^^^^

Concept
-------
The objective of this module is the adjustment of radar-based rainfall
estimates by rain gage observations. However, this module could also be
applied to adjust satellite rainfall by rain gage observations, remotely
sensed soil moisture patterns by ground truthing moisture sensors, or any
dense spatial point pattern which could be adjusted by sparse point
measurements (ground truth).

Basically, we only need two data sources:

- point observations (e.g. rain gage observations)
- set of (potentially irregular) unadjusted point values
  (e.g. remotely sensed rainfall)

:cite:`Goudenhoofdt2009` provide an excellent overview of adjustment
procedures. The general idea is that we quantify the error of the
remotely sensed rainfall at the rain gage locations, assuming the rain
gage observation to be accurate.

The error can be assumed to be purely additive
(:class:`~wradlib.adjust.AdjustAdd`), purely multiplicative
(:class:`~wradlib.adjust.AdjustMultiply`, :class:`~wradlib.adjust.AdjustMFB`)
or a mixture of both (:class:`~wradlib.adjust.AdjustMixed`).
If the error is assumed to be heterogeneous in space
(:class:`~wradlib.adjust.AdjustAdd`, :class:`~wradlib.adjust.AdjustMultiply`,
:class:`~wradlib.adjust.AdjustMixed`), the error at the rain gage locations is
interpolated to the radar bin locations and then used to adjust (correct)
the raw radar rainfall estimates. In case of the AdjustMFB approach, though,
the multiplicative error is assumed to be homogeneous in space.

Quick start
-----------
The basic procedure consists of creating an adjustment object from the class
you want to use for adjustment. After that, you can call the object with the
actual data that is to be adjusted. The following example is using the
additive error model with default settings. ``obs_coords`` and
``raw_coords`` represent arrays with coordinate pairs for the gage
observations and the radar bins, respectively. ``obs`` and ``raw`` are
arrays containing the actual data::

    adjuster = AdjustAdd(obs_coords, raw_coords)
    adjusted = adjuster(obs, raw)

Both ``obs`` and ``raw`` need to be flat (1-dimensional) arrays of shape (n,)
that have the same length as the the ``obs_coords`` and ``raw_coords`` arrays,
respectively.

The user can specify the approach that should be used to interpolate the error
in space, as well as the keyword arguments which control the behaviour of the
interpolation approach. For this purpose, all interpolation classes from the
:mod:`wradlib.ipol` module are available and can be passed by using the
``ipclass`` argument. The default interpolation class is
Inverse Distance Weighting (:class:`~wradlib.ipol.Idw`). If you want to use
e.g. linear barycentric interpolation::

    import wradlib.ipol as ipol
    adjuster = AdjustAdd(obs_coords, raw_coords, ipclass=ipol.Linear)
    adjusted = adjuster(obs, raw)

Warning
-------
    Be aware that there are a lot of control parameters that can dramatically
    influence the behaviour of the adjustment (which gauges are considered,
    how is an error interpolation carried out, ...). Read the docs carefully
    and try to experiment with the effects of the different control parameters.
    There might be situations in which the algorithms decides - based on the
    control parameter -  not to do an adjustment and just return the unadjusted
    values.

Cross validation
----------------
Another helpful feature is an easy-to-use method for leave-one-out
cross-validation :cite:`Cross-validation`. Cross validation is a standard
procedure for verifying rain gage adjustment or interpolation procedures. You
can start the cross validation in the same way as you start the actual
adjustment, however, you call the :meth:`~wradlib.adjust.AdjustBase.xvalidate`
method instead. The result of the cross validation are pairs of observation
and the corresponding adjustment result at the observation location. Using the
:mod:`wradlib.verify` module, you can compute error metrics for the cross
validation results::

    adjuster = AdjustAdd(obs_coords, raw_coords)
    observed, estimated = adjuster.xvalidate(obs, raw)
    from wradlib.verify import ErrorMetrics
    metrics = ErrorMetrics(observed, estimated)
    metrics.report()

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "AdjustBase",
    "AdjustMFB",
    "AdjustMultiply",
    "AdjustAdd",
    "AdjustMixed",
    "RawAtObs",
    "GageOnly",
    "AdjustNone",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
from scipy import spatial, stats

from wradlib import ipol, util


class AdjustBase(ipol.IpolBase):
    """The basic adjustment class that inherits to all other classes.

    All methods except the :meth:`~wradlib.adjust.AdjustBase.__call__` method
    are inherited to the following adjustment classes.

    Parameters
    ----------
    obs_coords : array of floats of shape (number of points, 2)
        x and y coordinate pairs of observation locations (e.g. rain gauges).
    raw_coords : array of floats of shape (number of points, 2)
        x and y coordinate pairs of raw (unadjusted) radar field
    nnear_raws : integer
        Defaults to 9. This parameter controls the number of radar bins or
        grid cells (in the neighbourhood of a rain gauge) which is used to
        compute the value of the radar observation AT a rain gauge.
    stat : string
        Defaults to 'median'. Must be either 'mean', 'median', or 'best'.
        This parameter controls the statistic that is used to compute the value
        of the radar observation AT a rain gauge based on the neighbourhood
        specified by parameter ``nnear_raws``.
    mingages : integer
        Defaults to 5. Minimum number of valid gages required for an
        adjustment. If less valid gauges are available, the adjustment
        procedure will return unadjusted raw values. If you do not want to use
        this feature, you need to set ``mingages=0``.
    minval : float
        If the gage or radar observation is below this threshold, the location
        will not be used for adjustment. For additive adjustment, this value
        should be set to zero (default value). For multiplicative adjustment,
        values larger than zero might be chosen in order to minimize
        artifacts.
    mfb_args : dictionary
        **Only used for AdjustMFB** - This set of parameters controls how the
        mean field bias is computed. Items of the dictionary are:

        - *method*: string
          defaults to 'linregr' which fits a regression line through observed
          and estimated values and than gets the bias from the inverse of
          the slope.
          Other values: 'mean' or 'median' compute the mean or the median of
          the ratios between gauge and radar observations.
        - *minslope*, *minr*, *maxp*:
          When using method='linregr', these parameters control whether a
          linear regression turned out to be robust (minimum allowable slope,
          minimum allowable correlation, maximim allowable p-value). If the
          regression result is not considered robust, no adjustment will
          take place.

    Ipclass : an interpolation class from :mod:`wradlib.ipol`
        **Not used for AdjustMFB** - default value is
        :class:`~wradlib.ipol.Idw` (Inverse Distance Weighting).
    ipargs : keyword arguments to create an instance of ipclass
        **Not used for AdjustMFB** - for :class:`~wradlib.ipol.Idw`, these
        keyword arguments would e.g. be ``nnear`` or ``p``.

    Examples
    --------

    See :ref:`/notebooks/multisensor/wradlib_adjust_example.ipynb`.

    """

    def __init__(
        self,
        obs_coords,
        raw_coords,
        nnear_raws=9,
        stat="median",
        mingages=5,
        minval=0.0,
        mfb_args=None,
        ipclass=ipol.Idw,
        **ipargs,
    ):

        # Check arguments
        if mfb_args is None:
            mfb_args = dict(method="linregr", minslope=0.1, minr=0.5, maxp=0.01)
        assert mfb_args["method"] in ["mean", "median", "linregr"], (
            "Argument mfb_args['method'] has to be one "
            "out of 'mean', 'median' or 'linregr'."
        )

        # These are the coordinates of the rain gage locations and
        # the radar bin locations
        self.obs_coords = self._make_coord_arrays(obs_coords)
        self.raw_coords = self._make_coord_arrays(raw_coords)

        # These are the general control parameters
        # for all adjustment procedures
        self.nnear_raws = nnear_raws
        self.stat = stat
        self.mingages = mingages
        self.minval = minval

        # Control parameters for specific adjustment procedures

        # for AdjustMFB
        self.mfb_args = mfb_args

        # interpolation class and its keyword arguments
        # ((needed for AdjustAdd, AdjustMultiply, AdjustMixed)
        self.ipclass = ipclass
        self.ipargs = ipargs
        # create a default instance of interpolator
        self.ip = ipclass(src=self.obs_coords, trg=self.raw_coords, **ipargs)

        # This method will quickly retrieve the actual radar values
        # at the gage locations
        self.get_raw_at_obs = RawAtObs(
            self.obs_coords, self.raw_coords, nnear=nnear_raws, stat=stat
        )

    def _checkip(self, ix, targets):
        """INTERNAL: Return a revised instance of the Interpolator class.

        When an instance of an Adjust... class is created, an instance of the
        desired
        Interpolation class (argument ipclass) is created as attribute
        *self.ip*). However, this instance is only valid in case all
        observation points (attribute *self.obs_coords*) have valid
        observation-radar pairs. In case points are missing (or in case the
        instance is called in the sourse of cross validation), a new instance
        has to be created which consideres the new constellation of
        observation-radar pairs.
        This method computes and returns this new instance.

        Parameters
        ----------
        ix : array of integers
            These are the indices of observation points with valid
            observation-radar pairs
        targets : array of floats of shape (number of target points, 2)
            Target coordinates for the interpolation

        Returns
        -------
        output : an instance of a class that inherited from
            wradlib.ipol.IpolBase

        """
        #    first, set interpolation targets (default: the radar coordinates)
        targets_default = False
        if targets is None:
            targets = self.raw_coords
            targets_default = True
        # second, compute inverse distance neighbours
        if (not len(ix) == len(self.obs_coords)) or (not targets_default):
            return self.ipclass(self.obs_coords[ix], targets, **self.ipargs)
        else:
            return self.ip

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Parameters
        ----------
        obs : flat (1-D) array of floats with shape (num gauges,)
            These are the gage observations used for adjustment. This array
            needs to be of the same length as the array "obs_coords" used to
            initialize the adjustment object.
        raw : flat (1-D) array of floats with shape (num radar cells,)
            These are the raw (unadjusted) radar rainfall values. This array
            needs to be of the same length as the array "raw_coords" used to
            initialize the adjustment object.
        targets : (INTERNAL - DO NOT USE)
            Array of floats. Coordinate pairs for locations on which the final
            adjustment product is interpolated
            Defaults to None. In this case, the output locations will be
            identical to the radar coordinates
        rawatobs : (INTERNAL - DO NOT USE)
            Array of floats. For internal use from AdjustBase.xvalidate only
            (defaults to None)
        ix : (INTERNAL - DO NOT USE)
            Array of integers. For internal use from AdjustBase.xvalidate only
            (defaults to None)

        """
        pass

    def _check_shape(self, obs, raw):
        """INTERNAL: Check consistency of the input data obs and raw with
        the shapes of the coordinates
        """
        # TODO
        pass

    def _get_valid_pairs(self, obs, raw):
        """INTERNAL: Helper method to identify valid obs-raw pairs"""
        # checking input shape consistency
        self._check_shape(obs, raw)
        # radar values at gage locations
        rawatobs = self.get_raw_at_obs(raw, obs)
        # check where both gage and radar observations are valid
        ix = np.intersect1d(
            util._idvalid(obs, minval=self.minval),
            util._idvalid(rawatobs, minval=self.minval),
        )
        return rawatobs, ix

    def xvalidate(self, obs, raw):
        """Leave-One-Out Cross Validation, applicable to all gage adjustment
        classes.

        This method will be inherited to other Adjust classes. It should thus
        be applicable to all adjustment procedures without any modification.
        This way, the actual adjustment procedure has only to be defined *once*
        in the :meth:`~wradlib.adjust.AdjustBase.__call__` method.

        The output of this method can be evaluated by using the
        `verify.ErrorMetrics` class.

        Parameters
        ----------
        obs : array of floats
        raw : array of floats

        Returns
        -------
        obs : array of floats
            valid observations at those locations which have a valid radar
            observation
        estatobs : array of floats
            estimated values at the valid observation locations

        """
        rawatobs, ix = self._get_valid_pairs(obs, raw)
        self.get_raws_directly_at_obs = RawAtObs(
            self.obs_coords, self.raw_coords, nnear=1
        )
        raws_directly_at_obs = self.get_raws_directly_at_obs(raw)
        ix = np.intersect1d(ix, util._idvalid(raws_directly_at_obs, minval=self.minval))
        # Container for estimation results at the observation location
        estatobs = np.zeros(obs.shape) * np.nan
        # check whether enough gages remain for adjustment
        if len(ix) <= (self.mingages - 1):
            # not enough gages for cross validation: return empty arrays
            return obs, estatobs
        # Now iterate over valid pairs
        for i in ix:
            # Pass all valid pairs except ONE which you pass as target
            ix_adjust = np.setdiff1d(ix, [i])
            estatobs[i] = self.__call__(
                obs,
                raws_directly_at_obs[i],
                self.obs_coords[i].reshape((1, -1)),
                rawatobs,
                ix_adjust,
            )
        return obs, estatobs


class AdjustAdd(AdjustBase):
    """Gage adjustment using an additive error model.

    First, an instance of AdjustAdd has to be created. Calling this instance
    then does the actual adjustment. The motivation behind this performance.
    In case the observation points are always the same for different time
    steps, the computation of neighbours and inverse distance weights only
    needs to be performed once.

    AdjustAdd automatically takes care of invalid gage or radar observations
    (e.g. NaN, Inf or other typical missing data flags such as -9999).
    However, in case e.g. the observation data contains missing values, the
    computation of the inverse distance weights needs to be repeated in
    :meth:`~wradlib.adjust.AdjustAdd.__call__` which is at the expense of
    performance.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :class:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of adjusted radar values

    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        # Get new Interpolator instance if necessary
        ip = self._checkip(ix, targets)

        # -----------------THIS IS THE ACTUAL ADJUSTMENT APPROACH--------------
        # The error is a difference
        error = obs[ix] - rawatobs[ix]
        # interpolate the error field
        iperror = ip(error)
        # add error field to raw and make sure no negatives occur
        return np.where((raw + iperror) < 0.0, 0.0, raw + iperror)


class AdjustMultiply(AdjustBase):
    """Gage adjustment using a multiplicative error model

    First, an instance of AdjustMultiply has to be created. Calling this
    instance then does the actual adjustment. The motivation behind this
    performance. In case the observation points are always the same for
    different time steps, the computation of neighbours and inverse distance
    weights only needs to be performed once during initialisation.

    AdjustMultiply automatically takes care of invalid gage or radar
    observations (e.g. NaN, Inf or other typical missing data flags such as
    -9999). However, in case e.g. the observation data contain missing values,
    the computation of the inverse distance weights needs to be repeated in
    :meth:`~wradlib.adjust.AdjustMultiply.__call__` which is at the expense of
    performance.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :meth:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of adjusted radar values

    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        # Get new Interpolator instance if necessary
        ip = self._checkip(ix, targets)

        # -----------------THIS IS THE ACTUAL ADJUSTMENT APPROACH--------------
        # computing the error
        error = obs[ix] / rawatobs[ix]
        # interpolate error field
        iperror = ip(error)
        # multiply error field with raw
        return iperror * raw


class AdjustMixed(AdjustBase):
    """Gage adjustment using a mixed error model (additive and multiplicative).

    The mixed error model assumes that you have both a multiplicative and an
    additive error term. The intention is to overcome the drawbacks of the
    purely additive and multiplicative approaches (see
    :class:`~wradlib.adjust.AdjustAdd` and
    :class:`~wradlib.adjust.AdjustMultiply`). The formal representation of the
    error model according to :cite:`Pfaff2010` is:

    .. math::

       R_{gage} = R_{radar} \\cdot (1 + \\delta) +0 \\epsilon

    :math:`\\delta` and :math:`\\epsilon` have to be assumed to be independent
    and normally distributed. The present implementation is based on a Least
    Squares estimation of :math:`\\delta` and :math:`\\epsilon` for each rain
    gage location. :math:`\\delta` and :math:`\\epsilon` are then interpolated
    and used to correct the radar rainfall field.

    The least squares implementation uses the equation for the error model plus
    the condition to minimize (:math:`\\delta^2 + \\epsilon^2`) for each gage
    location. The idea behind this is that :math:`\\epsilon` dominates the
    adjustment for small deviations between radar and gage while
    :math:`\\delta` dominates in case of large deviations.

    **Usage**:
    First, an instance of AdjustMixed has to be created. Calling this instance
    then does the actual adjustment. The motivation behind this is performance.
    In case the observation points are always the same for different time
    steps, the computation of neighbours and inverse distance weights only
    needs to be performed once during initialisation.

    AdjustMixed automatically takes care of invalid gage or radar observations
    (e.g. NaN, Inf or other typical missing data flags such as -9999).
    However, in case e.g. the observation data contain missing values, the
    computation of the inverse distance weights needs to be repeated in
    :func:`~wradlib.adjust.AdjustMixed.__call__` which is at the expense of
    performance.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :class:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of adjusted radar values



    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        # Get new Interpolator instance if necessary
        ip = self._checkip(ix, targets)

        # -----------------THIS IS THE ACTUAL ADJUSTMENT APPROACH--------------
        # computing epsilon and delta from least squares
        epsilon = (obs[ix] - rawatobs[ix]) / (rawatobs[ix] ** 2 + 1.0)
        delta = ((obs[ix] - epsilon) / rawatobs[ix]) - 1.0
        # interpolate error fields
        ipepsilon = ip(epsilon)
        ipdelta = ip(delta)
        # compute adjusted radar rainfall field
        return (1.0 + ipdelta) * raw + ipepsilon


class AdjustMFB(AdjustBase):
    """Multiplicative gage adjustment using *one* correction factor for the \
    entire domain.

    This method is also known as the Mean Field Bias correction.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :class:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of adjusted radar values

    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        # # Get new Interpolator instance if necessary
        # ip = self._checkip(ix, targets)

        # -----------------THIS IS THE ACTUAL ADJUSTMENT APPROACH--------------
        # compute ratios for each valid observation point
        ratios = np.ma.masked_invalid(obs[ix] / rawatobs.ravel()[ix])
        if len(np.where(np.logical_not(ratios.mask))[0]) < self.mingages:
            # Not enough valid pairs of raw and obs
            return raw
        if self.mfb_args["method"] == "mean":
            corrfact = np.mean(ratios)
        elif self.mfb_args["method"] == "median":
            corrfact = np.median(ratios)
        elif self.mfb_args["method"] == "linregr":
            corrfact = 1.0
            ix_ = np.where(np.logical_not(ratios.mask))[0]
            x = obs[ix][ix_]
            y = rawatobs[ix][ix_]
            # check whether we should adjust or not
            try:
                slope, intercept, r, p, stderr = stats.linregress(x, y)
            except Exception:
                slope, r, p = 0, 0, np.inf
            if (
                (slope > self.mfb_args["minslope"])
                and (r > self.mfb_args["minr"])
                and (p < self.mfb_args["maxp"])
            ):
                x = x[:, np.newaxis]
                try:
                    slope, _, _, _ = np.linalg.lstsq(x, y)
                    if not slope[0] == 0:
                        corrfact = 1.0 / slope[0]
                except Exception:
                    # no correction if linear regression fails
                    pass
        if type(corrfact) == np.ma.core.MaskedConstant:
            corrfact = 1.0
        return corrfact * raw


class AdjustNone(AdjustBase):
    """Same behaviour as the other adjustment classes, but simply returns the \
    unadjusted data.

    This class can be used for benchmark verification experiments as a control
    for unadjusted data.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :class:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of unadjusted radar values

    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        return raw


class GageOnly(AdjustBase):
    """Same behaviour as the other adjustment classes, but returns an \
    interpolation of rain gage observations

    First, an instance of GageOnly has to be created. Calling this instance
    then does the actual adjustment. The motivation behind this performance.
    In case the observation points are always the same for different time
    steps, the computation of neighbours and inverse distance weights only
    needs to be performed once during initialisation.

    GageOnly automatically takes care of invalid gage or radar observations
    (e.g. NaN, Inf or other typical missing data flags such as -9999).
    However, in case e.g. the observation data contain missing values, the
    computation of the inverse distance weights needs to be repeated in
    :func:`~wradlib.adjust.GageOnly.__call__` which is at the expense of
    performance.

    Note
    ----
    Inherits from :class:`wradlib.adjust.AdjustBase`

    For a complete overview of parameters for the initialisation of adjustment
    objects, as well as an extensive example, please see
    :class:`wradlib.adjust.AdjustBase`.

    Returns
    -------
    output : array of adjusted radar values
    """

    def __call__(self, obs, raw, targets=None, rawatobs=None, ix=None):
        """Returns an array of ``raw`` values that are adjusted by ``obs``.

        Calling an adjustment object works the same for all adjustment classes.
        Detailed instructions on the parameters ``obs`` and ``raw`` are
        provided in :meth:`wradlib.adjust.AdjustBase.__call__`.

        """
        # ----------------GENERIC PART FOR MOST __call__ methods---------------
        if (ix is None) or (rawatobs is None):
            # Check for valid observation-radar pairs in case this method has
            # not been called from self.xvalidate
            rawatobs, ix = self._get_valid_pairs(obs, raw)
        if len(ix) < self.mingages:
            # Not enough valid gages for adjustment? - return unadjusted data
            return raw
        # Get new Interpolator instance if necessary
        ip = self._checkip(ix, targets)

        # -----------------THIS IS THE ACTUAL ADJUSTMENT APPROACH--------------
        # interpolate gage observations
        return ip(obs[ix])


class RawAtObs:
    """Get the raw values in the neighbourhood of the observation points

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear: integer
        number of neighbours which should be considered in the vicinity of each
        point in obs
    stat: string
        function name

    """

    def __init__(self, obs_coords, raw_coords, nnear=9, stat="median"):
        self.statfunc = _get_statfunc(stat)
        self.raw_ix = _get_neighbours_ix(obs_coords, raw_coords, nnear)

    def __call__(self, raw, obs=None):
        """
        Returns the values of raw at the observation locations

        Parameters
        ----------
        raw : array of float
            raw values

        """
        # get the values of the raw neighbours of obs
        raw_neighbs = raw[self.raw_ix]
        # and summarize the values of these neighbours
        # by using a statistics option
        # (only needed in case nnear > 1, i.e. multiple neighbours
        # per observation location)
        if raw_neighbs.ndim > 1:
            return self.statfunc(obs, raw_neighbs)
        else:
            return raw_neighbs


def _get_neighbours_ix(obs_coords, raw_coords, nnear):
    """Returns ``nnear`` neighbour indices per ``obs_coords`` coordinate pair

    Parameters
    ----------
    obs_coords : array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    nnear : integer
        number of neighbours to be selected per coordinate pair of
        ``obs_coords``

    """
    # plant a tree
    tree = spatial.cKDTree(raw_coords)
    # return nearest neighbour indices
    return tree.query(obs_coords, k=nnear)[1]


def _get_statfunc(funcname):
    """Returns a function that corresponds to parameter ``funcname``

    Parameters
    ----------
    funcname : string
        a name of a numpy function OR another option known by _get_statfunc
        Potential options: 'mean', 'median', 'best'

    """
    try:
        # first try to find a numpy function which corresponds to <funcname>
        func = getattr(np, funcname)

        def newfunc(x, y):
            return func(y, axis=1)

    except Exception:
        # then try to find a function in this module with name funcname
        if funcname == "best":
            newfunc = best
        else:
            # if no function can be found, raise an Exception
            raise NameError("Unknown function name option: " + funcname)
    return newfunc


def best(x, y):
    """Find the values of y which corresponds best to x

    If x is an array, the comparison is carried out for each element of x

    Parameters
    ----------
    x : float or 1-d array of float
    y : array of float

    Returns
    -------
    output : 1-d array of float with length len(y)

    """
    if type(x) == np.ndarray:
        assert x.ndim == 1, "x must be a 1-d array of floats or a float."
        assert len(x) == len(y), "Length of x and y must be equal."
    if type(y) == np.ndarray:
        assert y.ndim <= 2, "y must be 1-d or 2-d array of floats."
    else:
        raise ValueError("y must be 1-d or 2-d array of floats.")
    x = np.array(x).reshape((-1, 1))
    if y.ndim == 1:
        y = np.array(y).reshape((1, -1))
        axis = None
    else:
        axis = 1
    return y[np.arange(len(y)), np.argmin(np.abs(x - y), axis=axis)]


if __name__ == "__main__":
    print("wradlib: Calling module <adjust> as main...")
