#-------------------------------------------------------------------------------
# Name:         adjust
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Adjustment
^^^^^^^^^^

Adjusting remotely sensed spatial data by ground truth (gage observations)

The main objective of this module is the adjustment of radar-based QPE
by rain gage observations. However, this module can also be applied to adjust
satellite rainfall by rain gage observations, remotely sensed soil moisture
patterns by ground truthing moisture sensors or any spatial point pattern
which could be adjusted by selected point measurements (ground truth).

Basically, we only need two data sources:

- point observations (e.g. rain gage observations)

- set of (potentially irregular) unadjusted point values (e.g. remotely sensed rainfall)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   AdjustMFB
   AdjustMultiply
   AdjustAdd
   AdjustMixed
   Raw_at_obs

"""

# site packages
import numpy as np
from scipy.spatial import cKDTree

# wradlib modules
import wradlib.ipol as ipol
import wradlib.util as util


class AdjustBase(ipol.IpolBase):
    """
    The basic adjustment class

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.
    mingages : integer
        minimum number of gages which are required for an adjustment

    """
    def __init__(self, obs_coords, raw_coords, nnear_raws=9, stat='median', nnear_idw=6, p_idw=2., mingages=5):
        self.obs_coords = self._make_coord_arrays(obs_coords)
        self.raw_coords = self._make_coord_arrays(raw_coords)
        self.nnear_raws = nnear_raws
        self.stat       = stat
        self.nnear_idw  = nnear_idw
        self.p_idw      = p_idw
        self.mingages   = mingages
        self.get_raw_at_obs = Raw_at_obs(self.obs_coords,  self.raw_coords, nnear=nnear_raws, stat=stat)
        self.ip = ipol.Idw(src=self.obs_coords, trg=self.raw_coords, nnearest=nnear_idw, p=p_idw)
    def _check_shape(self, obs, raw):
        """
        Check consistency of the input data obs and raw with the shapes of the coordinates
        """
        print 'TODO WARNING: fill in _check_shape method'


class AdjustAdd(AdjustBase):
    """Gage adjustment using an additive error model

    First, an instance of AdjustAdd has to be created. Calling this instance then
    does the actual adjustment. The motivation behind this performance. In case
    the observation points are always the same for different time steps, the computation
    of neighbours and invserse distance weights only needs to be performed once.

    AdjustAdd automatically takes care of invalid gage or radar observations (e.g.
    NaN, Inf or other typical missing data flags such as -9999. However, in case
    e.g. the observation data contain missing values, the computation of the inverse
    distance weights needs to be repeated in __call__ which is at the expense of
    performance.

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.
    mingages : integer
        minimum number of gages which are required for an adjustment

    Returns
    -------
    output : array of adjusted radar values

    Examples
    --------
    >>> import wradlib.adjust as adjust
    >>> import numpy as np
    >>> import pylab as pl
    >>> # 1-d example including all available adjustment methods
    >>> # --------------------------------------------------------------------------
    >>> # gage and radar coordinates
    >>> obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    >>> radar_coords = np.arange(0,101)
    >>> # true rainfall
    >>> truth = np.abs(np.sin(0.1*radar_coords))
    >>> # radar error
    >>> erroradd = np.random.uniform(0,0.5,len(radar_coords))
    >>> errormult= 1.1
    >>> # radar observation
    >>> radar = errormult*truth + erroradd
    >>> # gage observations are assumed to be perfect
    >>> obs = truth[obs_coords]
    >>> # add a missing value to observations (just for testing)
    >>> obs[1] = np.nan
    >>> # adjust the radar observation by additive model
    >>> add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=1)
    >>> add_adjusted = add_adjuster(obs, radar)
    >>> # adjust the radar observation by multiplicative model
    >>> mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=1)
    >>> mult_adjusted = mult_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by MFB
    >>> mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=1)
    >>> mfb_adjusted = mfb_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by AdjustMixed
    >>> mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=1)
    >>> mixed_adjusted = mixed_adjuster(obs, radar)
    >>> line1 = pl.plot(radar_coords, radar, 'k-', label="raw radar")
    >>> line2 = pl.plot(obs_coords, obs, 'ro', label="gage obs")
    >>> line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="adjusted by AdjustAdd")
    >>> line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="adjusted by AdjustMultiply")
    >>> line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="adjusted by AdjustMFB")
    >>> line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="adjusted by AdjustMixed")
    >>> pl.legend()
    >>> pl.show()

    Notes
    -----
    Inherits from AdjustBase

    """

    def __call__(self, obs, raw):
        """
        Return the field of raw values adjusted by obs

        Parameters
        ----------
        obs : array of float
            observations
        raw : array of float
            raw unadjusted field

        """
        # checking input shape consistency
        self._check_shape(obs, raw)
        # radar values at gage locations
        rawatobs = self.get_raw_at_obs(raw, obs)
        # check where both gage and radar observations are valid
        ix = np.intersect1d( util._idvalid(obs),  util._idvalid(rawatobs))
        # check whether enough gages remain for adjustment
        if len(ix)<=self.mingages:
            # no adjustment
            print "Not enough gages for adjustment...returning unadjusted data."
            return raw
        # computing the error
        error = obs[ix] - rawatobs[ix]
        # if not all locations have valid values, we need to recalculate the inverse distance neighbours
        if not len(ix)==len(obs):
            print "Missing values: Need to recalculate inverse distance weights..."
            ip = ipol.Idw(src=self.obs_coords[ix], trg=self.raw_coords, nnearest=self.nnear_idw, p=self.p_idw)
        else:
            ip = self.ip
        # interpolate error field
        iperror = ip(error)
        # add error field to raw and cut negatives to zero
        return np.where( (raw + iperror)<0., 0., raw + iperror)


class AdjustMultiply(AdjustBase):
    """Gage adjustment using a multiplicative error model

    First, an instance of AdjustMultiply has to be created. Calling this instance then
    does the actual adjustment. The motivation behind this performance. In case
    the observation points are always the same for different time steps, the computation
    of neighbours and invserse distance weights only needs to be performed once during
    initialisation.

    AdjustMultiply automatically takes care of invalid gage or radar observations (e.g.
    NaN, Inf or other typical missing data flags such as -9999. However, in case
    e.g. the observation data contain missing values, the computation of the inverse
    distance weights needs to be repeated in __call__ which is at the expense of
    performance.

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.
    mingages : integer
        minimum number of gages which are required for an adjustment

    Returns
    -------
    output : array of adjusted radar values

    Examples
    --------
    >>> import wradlib.adjust as adjust
    >>> import numpy as np
    >>> import pylab as pl
    >>> # 1-d example including all available adjustment methods
    >>> # --------------------------------------------------------------------------
    >>> # gage and radar coordinates
    >>> obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    >>> radar_coords = np.arange(0,101)
    >>> # true rainfall
    >>> truth = np.abs(np.sin(0.1*radar_coords))
    >>> # radar error
    >>> erroradd = np.random.uniform(0,0.5,len(radar_coords))
    >>> errormult= 1.1
    >>> # radar observation
    >>> radar = errormult*truth + erroradd
    >>> # gage observations are assumed to be perfect
    >>> obs = truth[obs_coords]
    >>> # add a missing value to observations (just for testing)
    >>> obs[1] = np.nan
    >>> # adjust the radar observation by additive model
    >>> add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=1)
    >>> add_adjusted = add_adjuster(obs, radar)
    >>> # adjust the radar observation by multiplicative model
    >>> mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=1)
    >>> mult_adjusted = mult_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by MFB
    >>> mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=1)
    >>> mfb_adjusted = mfb_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by AdjustMixed
    >>> mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=1)
    >>> mixed_adjusted = mixed_adjuster(obs, radar)
    >>> line1 = pl.plot(radar_coords, radar, 'k-', label="raw radar")
    >>> line2 = pl.plot(obs_coords, obs, 'ro', label="gage obs")
    >>> line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="adjusted by AdjustAdd")
    >>> line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="adjusted by AdjustMultiply")
    >>> line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="adjusted by AdjustMFB")
    >>> line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="adjusted by AdjustMixed")
    >>> pl.legend()
    >>> pl.show()

    Notes
    -----
    Inherits from AdjustBase

    """

    def __call__(self, obs, raw, minval):
        """
        Returns the field of raw values adjusted by obs

        Parameters
        ----------
        obs : array of float
            observations
        raw : array of float
            raw unadjusted field
        minval : float
            if the gage or radar observation is below this threshold, the location
            will not be used for adjustment in order to avoid extreme ratios

        """
        # checking input shape consistency
        self._check_shape(obs, raw)
        # radar values at gage locations
        rawatobs = self.get_raw_at_obs(raw, obs)
        # check where both gage and radar observations are valid
        ix = np.intersect1d( util._idvalid(obs, minval=minval),  util._idvalid(rawatobs, minval=minval))
        # check whether enough gages remain for adjustment
        if len(ix)<=self.mingages:
            # no adjustment
            print "Not enough gages for adjustment...returning unadjusted data."
            return raw
        # computing the error
        error = obs[ix] / rawatobs[ix]
        # if not all locations have valid values, we need to recalculate the inverse distance neighbours
        if not len(ix)==len(obs):
            ip = ipol.Idw(src=self.obs_coords[ix], trg=self.raw_coords, nnearest=self.nnear_idw, p=self.p_idw)
        else:
            ip = self.ip
        # interpolate error field
        iperror = ip(error)
        # multiply error field with raw
        return iperror * raw


class AdjustMixed(AdjustBase):
    """Gage adjustment using a mixed error model (additive and multiplicative).

    The mixed error model assumes that you have both a multiplicative and an
    additive error term. The intention is to overcome the drawbacks of the purely
    additive and multiplicative approaches (see AdjustAdd and AdjustMultiply). The
    formal reprentation of the error model according to [Pfaff2010]_ is:

    R(gage) = R(radar) * (1+delta) + epsilon

    delta and epsilon have to be assumed to be independent and normally distributed.
    The present implementation is based on a Least Squares estimation of delta and
    epsilon for each rain gage location. delta and epsilon are then interpolated
    and used to correct the radar rainfall field. The least squares implementation
    uses the equation for the error model plus the condition to minimize
    (delta**2 + epsilon**2) for each gage location. The idea behind this is that epsilon
    dominates the adjustment for small deviations between radar and gage while
    delta dominates in case of large deviations.

    **Usage**: First, an instance of AdjustMMixed has to be created. Calling this instance then
    does the actual adjustment. The motivation behind this is performance. In case
    the observation points are always the same for different time steps, the computation
    of neighbours and invserse distance weights only needs to be performed once during
    initialisation.

    AdjustMixed automatically takes care of invalid gage or radar observations (e.g.
    NaN, Inf or other typical missing data flags such as -9999. However, in case
    e.g. the observation data contain missing values, the computation of the inverse
    distance weights needs to be repeated in __call__ which is at the expense of
    performance.

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.
    mingages : integer
        minimum number of gages which are required for an adjustment

    Returns
    -------
    output : array of adjusted radar values

    Examples
    --------
    >>> import wradlib.adjust as adjust
    >>> import numpy as np
    >>> import pylab as pl
    >>> # 1-d example including all available adjustment methods
    >>> # --------------------------------------------------------------------------
    >>> # gage and radar coordinates
    >>> obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    >>> radar_coords = np.arange(0,101)
    >>> # true rainfall
    >>> truth = np.abs(np.sin(0.1*radar_coords))
    >>> # radar error
    >>> erroradd = np.random.uniform(0,0.5,len(radar_coords))
    >>> errormult= 1.1
    >>> # radar observation
    >>> radar = errormult*truth + erroradd
    >>> # gage observations are assumed to be perfect
    >>> obs = truth[obs_coords]
    >>> # add a missing value to observations (just for testing)
    >>> obs[1] = np.nan
    >>> # adjust the radar observation by additive model
    >>> add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=1)
    >>> add_adjusted = add_adjuster(obs, radar)
    >>> # adjust the radar observation by multiplicative model
    >>> mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=1)
    >>> mult_adjusted = mult_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by MFB
    >>> mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=1)
    >>> mfb_adjusted = mfb_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by AdjustMixed
    >>> mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=1)
    >>> mixed_adjusted = mixed_adjuster(obs, radar)
    >>> line1 = pl.plot(radar_coords, radar, 'k-', label="raw radar")
    >>> line2 = pl.plot(obs_coords, obs, 'ro', label="gage obs")
    >>> line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="adjusted by AdjustAdd")
    >>> line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="adjusted by AdjustMultiply")
    >>> line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="adjusted by AdjustMFB")
    >>> line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="adjusted by AdjustMixed")
    >>> pl.legend()
    >>> pl.show()

    Notes
    -----
    Inherits from AdjustBase

    References
    ----------
    .. [Pfaff2010] Pfaff, T., 2010. Radargestuetzte Schaetzung von Niederschlagsensembles (in German).
        In: Bronstert et al. (Eds.). Operationelle Abfluss- und Hochwasservorhersage in Quellgebieten.
        Final Project Report, pp. 113-118. URL: http://www.rimax-hochwasser.de/fileadmin/user_uploads/RIMAX_PUB_22_0015_Abschlussbericht%20OPAQUE_final.pdf.

    """

    def __call__(self, obs, raw):
        """
        Returns the field of raw values adjusted by obs

        Parameters
        ----------
        obs : array of float
            observations
        raw : array of float
            raw unadjusted field

        """
        # checking input shape consistency
        self._check_shape(obs, raw)
        # radar values at gage locations
        rawatobs = self.get_raw_at_obs(raw, obs)
        # check where both gage and radar observations are valid
        ix = np.intersect1d( util._idvalid(obs),  util._idvalid(rawatobs))
        # check whether enough gages remain for adjustment
        if len(ix)<=self.mingages:
            # no adjustment
            print "Not enough gages for adjustment...returning unadjusted data."
            return raw
        # computing epsilon and delta from least squares
        epsilon = (obs[ix] - rawatobs[ix]) / (rawatobs[ix]**2 + 1.)
        delta   = ( (obs[ix] - epsilon) / rawatobs[ix] ) - 1.
        # if not all locations have valid values, we need to recalculate the inverse distance neighbours
        if not len(ix)==len(obs):
            ip = ipol.Idw(src=self.obs_coords[ix], trg=self.raw_coords, nnearest=self.nnear_idw, p=self.p_idw)
        else:
            ip = self.ip
        # interpolate error fields
        ipepsilon = ip(epsilon)
        ipdelta = ip(delta)
        # compute adjusted radar rainfall field
        return (1. + ipdelta) * raw + ipepsilon



class AdjustMFB(AdjustBase):
    """
    Multiplicative gage adjustment using *one* correction factor for the entire domain

    This method is also known as the Mean Field Bias correction

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    mingages : integer
        minimum number of gages which are required for an adjustment

    Returns
    -------
    output : array of adjusted radar values

    Examples
    --------
    >>> import wradlib.adjust as adjust
    >>> import numpy as np
    >>> import pylab as pl
    >>> # 1-d example including all available adjustment methods
    >>> # --------------------------------------------------------------------------
    >>> # gage and radar coordinates
    >>> obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    >>> radar_coords = np.arange(0,101)
    >>> # true rainfall
    >>> truth = np.abs(np.sin(0.1*radar_coords))
    >>> # radar error
    >>> erroradd = np.random.uniform(0,0.5,len(radar_coords))
    >>> errormult= 1.1
    >>> # radar observation
    >>> radar = errormult*truth + erroradd
    >>> # gage observations are assumed to be perfect
    >>> obs = truth[obs_coords]
    >>> # add a missing value to observations (just for testing)
    >>> obs[1] = np.nan
    >>> # adjust the radar observation by additive model
    >>> add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=1)
    >>> add_adjusted = add_adjuster(obs, radar)
    >>> # adjust the radar observation by multiplicative model
    >>> mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=1)
    >>> mult_adjusted = mult_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by MFB
    >>> mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=1)
    >>> mfb_adjusted = mfb_adjuster(obs, radar,0.)
    >>> # adjust the radar observation by AdjustMixed
    >>> mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=1)
    >>> mixed_adjusted = mixed_adjuster(obs, radar)
    >>> line1 = pl.plot(radar_coords, radar, 'k-', label="raw radar")
    >>> line2 = pl.plot(obs_coords, obs, 'ro', label="gage obs")
    >>> line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="adjusted by AdjustAdd")
    >>> line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="adjusted by AdjustMultiply")
    >>> line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="adjusted by AdjustMFB")
    >>> line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="adjusted by AdjustMixed")
    >>> pl.legend()
    >>> pl.show()

    Notes
    -----
    Inherits from AdjustBase

    """

    def __call__(self, obs, raw, minval):
        """
        Return the field of raw values adjusted by obs

        Parameters
        ----------
        obs : array of float
            observations
        raw : array of float
            raw unadjusted field
        minval : float
            if the gage or radar observation is below this threshold, the location
            will not be used for calculating the mean field bias

        """
        # checking input shape consistency
        self._check_shape(obs, raw)
        # computing the multiplicative error for points of significant rainfall
        rawatobs = self.get_raw_at_obs(raw, obs)
        # check where both gage and radar observations are valid
        ix = np.intersect1d( util._idvalid(obs, minval=minval),  util._idvalid(rawatobs, minval=minval))
        # check if there are enough remaining gages for adjustment
        if len(ix)<=self.mingages:
            # no adjustment
            print "Not enough gages for adjustment...returning unadjusted data."
            return raw
        ratios = obs[ix] / rawatobs[ix]
        # compute adjustment factor
        thesum = np.nansum(ratios)
        num    = len(ratios) - len(np.where(np.isnan(ratios))[0])
        if (not np.isnan(thesum)) and (not num==0):
            corrfact = thesum / num
        else:
            corrfact = 1
        # return adjusted data
        return corrfact*raw


class Raw_at_obs():
    """
    Get the raw values in the neighbourhood of the observation points

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear: integer
        number of neighbours which should be considered in the vicinity of each point in obs
    stat: string
        function name

    """
    def __init__(self, obs_coords, raw_coords, nnear=9, stat='median'):
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
        # and summarize the values of these neighbours by using a statistics option
        # (only needed in case nnear > 1, i.e. multiple neighnours per observation location)
        if raw_neighbs.ndim > 1:
            return self.statfunc(obs, raw_neighbs)
        else:
            return raw_neighbs


def _get_neighbours_ix(obs_coords, raw_coords, nnear):
    """
    Returns <nnear> neighbour indices per <obs_coords> coordinate pair

    Parameters
    ----------
    obs_coords : array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    nnear : integer
        number of neighbours to be selected per coordinate pair of obs_coords

    """
    # plant a tree
    tree = cKDTree(raw_coords)
    # return nearest neighbour indices
    return tree.query(obs_coords, k=nnear)[1]


def _get_neighbours(obs_coords, raw_coords, raw, nnear):
    """
    Returns <nnear> neighbour values per <obs_coords> coordinate pair

    Parameters
    ----------
    obs_coords : array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    raw : array of float of shape (num_points,...)
        this is the data corresponding to the coordinate pairs raw_coords
    nnear : integer
        number of neighbours to be selected per coordinate pair of obs_coords

    """
    # plant a tree
    tree = cKDTree(raw_coords)
    # retrieve nearest neighbour indices
    ix = tree.query(obs_coords, k=nnear)[1]
    # return the values of the nearest neighbours
    return raw[ix]


def _get_statfunc(funcname):
    """
    Returns a function that corresponds to parameter <funcname>

    Parameters
    ----------
    funcname : string
        a name of a numpy function OR another option known by _get_statfunc
        Potential options: 'mean', 'median', 'best'

    """
    try:
        # first try to find a numpy function which corresponds to <funcname>
        func = getattr(np,funcname)
        def newfunc(x, y):
            return func(y, axis=1)
    except:
        try:
            # then try to find a function in this module with name funcname
            if funcname=='best':
                newfunc=best
        except:
            # if no function can be found, raise an Exception
            raise NameError('Unkown function name option: '+funcname)
    return newfunc


def best(x, y):
    """
    Find the values of y which corresponds best to x

    If x is an array, the comparison is carried out for each element of x

    Parameters
    ----------
    x : float or 1-d array of float
    y : array of float

    Returns
    -------
    output : 1-d array of float with length len(y)

    """
    if type(x)==np.ndarray:
        assert x.ndim==1, 'x must be a 1-d array of floats or a float.'
        assert len(x)==len(y), 'Length of x and y must be equal.'
    if type(y)==np.ndarray:
        assert y.ndim<=2, 'y must be 1-d or 2-d array of floats.'
    else:
        raise ValueError('y must be 1-d or 2-d array of floats.')
    x = np.array(x).reshape((-1,1))
    if y.ndim==1:
        y = np.array(y).reshape((1,-1))
        axis = None
    else:
        axis = 1
    return y[np.arange(len(y)),np.argmin(np.abs(x-y), axis=axis)]


def get_raw_at_obs(obs_coords, raw_coords, obs, raw, nnear=9, stat='median'):
    """
    Get the raw values in the neighbourhood of the observation points

    Parameters
    ----------

    obs_coords :

    raw: Datset of raw values (which shall be adjusted by obs)
    nnear: number of neighbours which should be considered in the vicinity of each point in obs
    stat: a numpy statistical function which should be used to summarize the values of raw in the neighbourshood of obs

    """
    # get the values of the raw neighbours of obs
    raw_neighbs = _get_neighbours(obs_coords, raw_coords, raw, nnear)
    # and summarize the values of these neighbours by using a statistics option
    return _get_statfunc(stat)(raw_neighbs)




if __name__ == '__main__':
    print 'wradlib: Calling module <adjust> as main...'
