# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        util
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not attributable
to the other modules

.. autosummary::
   :nosignatures:
   :toctree: generated/

   aggregate_in_time
   from_to

"""
import numpy as np
import datetime as dt
from time import mktime
from scipy import interpolate
from scipy.spatial import cKDTree



def aggregate_in_time(src, dt_src, dt_trg, taxis=0, func='sum'):
    """Aggregate time series data to a coarser temporal resolution.

    Parameters
    ----------

    src : array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The position
        of the time dimension is indicated by the *taxis* argument. The number
        of time steps corresponds to the length of the time dimension.

    taxis : integer
        This is the position of the time dimension in array *src*.

    dt_src : array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is the
        upper limit of time step 1 and the lower limit of time step 2 and so on.

    dt_trg : array of datetime objects
        Must be of length *number of output time steps + 1* analogously to dt_src.
        This means: dt_trg[0] is the lower limit of output time step 1, dt_trg[1]
        is the upper limit of output time step 1 and the lower limit of output
        time step 2 and so on.

    func : numpy function name, e.g. 'sum', 'mean'
        Defines the way the data should be aggregated. The string must correspond
        to a valid numpy function, e.g. 'sum', 'mean', 'min', 'max'.

    Returns
    -------

    output : array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the array
        *dt_trg* which defines the limits of the output time step intervals.

    Examples
    --------
    >>> src = np.arange(8*4).reshape( (8,4) )
    >>> print 'source time series:'
    >>> print src
    >>> dt_src = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(hours=i) for i in range(9) ]
    >>> print 'source time interval limits:'
    >>> for tim in dt_src: print tim
    >>> print 'target time interval limits:'
    >>> dt_trg = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(seconds=i*3600*4) for i in range(4) ]
    >>> for tim in dt_trg: print tim
    >>> print 'target time series'
    >>> print aggregate_in_time(src, dt_src, dt_trg, axis=0, func='sum')


    """
##    src, dt_src, dt_trg = np.array(src), np.array(dt_src), np.array(dt_trg)
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)
    trg_shape = list(src.shape)
    trg_shape[taxis] = len(dt_trg)-1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)
    for i in range(len(dt_trg)-1):
        trg_slice = [slice(0,j) for j in trg.shape]
        trg_slice[taxis] = i
        src_slice = [slice(0,src.shape[j]) for j in range(len(src.shape))]
        src_slice[taxis] = np.where( np.logical_and(dt_src<=dt_trg[i+1], dt_src>=dt_trg[i]) )[0][:-1]
        if len(src_slice[taxis])==0:
            trg[trg_slice] = np.nan
        else:
            trg[trg_slice] = _get_func(func)(src[tuple(src_slice)], axis=taxis)
    return trg

def sum_over_time_windows(src, dt_src, dt_trg, minpercvalid):
    """Returns the sums of time series <src> within the time windows dt_trg
    """
    assert len(src)+1==len(dt_src), "Length of time series array <src> must be one less than datetime array <dt_src>."
    try:
        dt_src = [dt.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S") for dtime in dt_src]
    except TypeError:
        pass
    try:
        dt_trg = [dt.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S") for dtime in dt_trg]
    except TypeError:
        pass

    accum = np.repeat(np.nan, len(dt_trg)-1)

    for i, tstart in enumerate(dt_trg[:-1]):
        tend = dt_trg[i+1]
        # accumulate gage data to target time windows
        ix = np.where((dt_src>tstart) & (dt_src <= tend))[0] - 1
        if len(src[ix])==0:
            continue
        elif len(np.where(np.isnan( src[ix] ))[0]) / len(src[ix]) < minpercvalid/100.:
            accum[i] = np.nansum( src[ix] )
    return accum


def mean_over_time_windows(src, dt_src, dt_trg, minbasepoints=1):
    """UNDER DEVELOPMENT: Aggregate time series data to a coarser temporal resolution.

    Parameters
    ----------

    src : array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The position
        of the time dimension is indicated by the *taxis* argument. The number
        of time steps corresponds to the length of the time dimension.

    taxis : integer
        This is the position of the time dimension in array *src*.

    dt_src : array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is the
        upper limit of time step 1 and the lower limit of time step 2 and so on.

    dt_trg : array of datetime objects
        Must be of length *number of output time steps + 1* analogously to dt_src.
        This means: dt_trg[0] is the lower limit of output time step 1, dt_trg[1]
        is the upper limit of output time step 1 and the lower limit of output
        time step 2 and so on.

    func : numpy function name, e.g. 'sum', 'mean'
        Defines the way the data should be aggregated. The string must correspond
        to a valid numpy function, e.g. 'sum', 'mean', 'min', 'max'.

    Returns
    -------

    output : array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the array
        *dt_trg* which defines the limits of the output time step intervals.

    Examples
    --------
    >>> src = np.arange(8*4).reshape( (8,4) )
    >>> print 'source time series:'
    >>> print src
    >>> dt_src = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(hours=i) for i in range(9) ]
    >>> print 'source time interval limits:'
    >>> for tim in dt_src: print tim
    >>> print 'target time interval limits:'
    >>> dt_trg = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(seconds=i*3600*4) for i in range(4) ]
    >>> for tim in dt_trg: print tim
    >>> print 'target time series'
    >>> print aggregate_in_time(src, dt_src, dt_trg, axis=0, func='sum')


    """
    # Convert input time steps to numpy arrays
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)
    # Create a new container for the target data
    trg_shape = list(src.shape)
    trg_shape[0] = len(dt_trg)-1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)
    for i in range(len(dt_trg)-1):
        # width of window
        width = float(_tdelta2seconds( dt_trg[i+1] - dt_trg[i] ))
        # These are the intervals completely INSIDE the target time window
        src_ix = np.where(np.logical_and(dt_src > dt_trg[i], dt_src < dt_trg[i+1]))[0]
        intervals = dt_src[src_ix[1:]] - dt_src[src_ix[:-1]]
        # check left edge
        intervals = np.insert(intervals, 0, dt_src[src_ix[0]] - dt_trg[i])
        if src_ix[0]>0:
            src_ix = np.insert(src_ix, 0, src_ix[0]-1)
        # check right edge
        intervals = np.append(intervals, dt_trg[i+1] - dt_src[src_ix[-1]])
        if src_ix[-1]>(len(dt_src)-1):
            src_ix = np.append(src_ix, src_ix[-1]+1)
        # convert to seconds
        intervals = np.array([_tdelta2seconds(interval) for interval in intervals])
        # compute weights
        weights = intervals / width
        # compute weighted mean
        trg[i] = np.dot(np.transpose(src[src_ix]), weights)
    return trg


def average_over_time_windows(src, dt_src, dt_trg, maxdist=3600, helper_interval=300, **ipargs):
    """UNDER DEVELOPMENT: Computes the average of a time series over given time windows.

    This function computes the average values of an irregular time series ``src``
    within given time windows ``dt_trg``. The datetimes of the original time series
    are given by ``dt_src``. The idea of this function is to create regular helper
    timesteps at an interval length given by ``helper_interval``. The values of
    ``src`` are then interpolated to these helper time steps, and the resulting
    helper values are finally averaged over the given target time windows.


    Parameters
    ----------

    src : array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The position
        of the time dimension is indicated by the *taxis* argument. The number
        of time steps corresponds to the length of the time dimension.

    taxis : integer
        This is the position of the time dimension in array *src*.

    dt_src : array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is the
        upper limit of time step 1 and the lower limit of time step 2 and so on.

    dt_trg : array of datetime objects
        Must be of length *number of output time steps + 1* analogously to dt_src.
        This means: dt_trg[0] is the lower limit of output time step 1, dt_trg[1]
        is the upper limit of output time step 1 and the lower limit of output
        time step 2 and so on.

    func : numpy function name, e.g. 'sum', 'mean'
        Defines the way the data should be aggregated. The string must correspond
        to a valid numpy function, e.g. 'sum', 'mean', 'min', 'max'.

    Returns
    -------

    output : array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the array
        *dt_trg* which defines the limits of the output time step intervals.

    Examples
    --------
    >>> src = np.arange(8*4).reshape( (8,4) )
    >>> print 'source time series:'
    >>> print src
    >>> dt_src = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(hours=i) for i in range(9) ]
    >>> print 'source time interval limits:'
    >>> for tim in dt_src: print tim
    >>> print 'target time interval limits:'
    >>> dt_trg = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + dt.timedelta(seconds=i*3600*4) for i in range(4) ]
    >>> for tim in dt_trg: print tim
    >>> print 'target time series'
    >>> print aggregate_in_time(src, dt_src, dt_trg, axis=0, func='sum')


    """
    # Convert input time steps to numpy arrays
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)

    trg_secs = np.array([mktime(tstep.timetuple()) for tstep in dt_trg])
    src_secs = np.array([mktime(tstep.timetuple()) for tstep in dt_src])
    helper_secs = np.arange(trg_secs[0],trg_secs[-1],helper_interval)

    # Interpolate to target points
    f = interpolate.interp1d(src_secs, src, axis=0, bounds_error=False)
    helpers = f(helper_secs)

    # Mask those values as invalid which are more than maxdist from the next source point
    tree = cKDTree(src_secs.reshape((-1,1)))
    dists, ix = tree.query(helper_secs.reshape((-1,1)), k=1)
    # deal with edges (in case of extrapolation, we apply nearest neighbour)
    np.where(np.isnan(helpers), src[ix], helpers)
    # mask out points which are to far from the next source point
    helpers[np.where(dists>maxdist)[0]] = np.nan

    # Create a new container for the target data
    trg_shape = list(src.shape)
    trg_shape[0] = len(dt_trg)-1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)

    for i in range(len(dt_trg)-1):
        # width of window
        width = float(_tdelta2seconds( dt_trg[i+1] - dt_trg[i] ))
        # These are the intervals completely INSIDE the target time window
        helper_ix = np.where(np.logical_and(dt_src >= dt_trg[i], dt_src <= dt_trg[i+1]))[0]
        trg[i] = np.mean(helpers[helper_ix], axis=0)

    return trg



def _get_func(funcname):
    """
    Retrieve the numpy function with name <funcname>

    Parameters
    ----------
    funcname : string

    """
    try:
        func = getattr(np,funcname)
    except:
        print '<'+funcname+'> is not a valid function in numpy...'
        raise
    return func


def _shape2size(shape):
    """
    Compute the size which corresponds to a shape
    """
    out=1
    for item in shape:
        out*=item
    return out


def from_to(tstart, tend, tdelta):
    """Return a list of timesteps from <tstart> to <tend> of length <tdelta>

    Parameters
    ----------
    tstart : datetime isostring (%Y%m%d %H:%M:%S), e.g. 2000-01-01 15:34:12
        or datetime object
    tend : datetime isostring (%Y%m%d %H:%M:%S), e.g. 2000-01-01 15:34:12
        or datetime object
    tdelta : integer representing time interval in SECONDS

    Returns
    -------
    output : list of datetime.datetime objects

    """
    if not type(tstart)==dt.datetime:
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not type(tend)==dt.datetime:
        tend   = dt.datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    tdelta  = dt.timedelta(seconds=tdelta)
    tsteps = [tstart,]
    tmptime = tstart
    while True:
        tmptime = tmptime + tdelta
        if tmptime > tend:
            break
        else:
            tsteps.append(tmptime)
    return tsteps

def _tdelta2seconds(tdelta):
    """
    Convert a dt.timedelta object to seconds

    Parameters
    ----------
    tdelta : a dt.timedelta object

    """
    return tdelta.days * 86400 + tdelta.seconds

def _get_tdelta(tstart, tend, as_secs=False):
    """Returns the difference between two datetimes
    """
    if not type(tstart)==dt.datetime:
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not type(tend)==dt.datetime:
        tend   = dt.datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    if not as_secs:
        return tend-tstart
    else:
        return _tdelta2seconds(tend-tstart)



def _idvalid(data, isinvalid=[-99., 99, -9999., -9999], minval=None, maxval=None):
    """Identifies valid entries in an array and returns the corresponding indices

    Invalid values are NaN and Inf. Other invalid values can be passed using the
    isinvalid keyword argument.

    Parameters
    ----------
    data : array of floats
    invalid : list of what is considered an invalid value

    """
    ix = np.ma.masked_invalid(data).mask
    for el in isinvalid:
        ix = np.logical_or(ix, np.ma.masked_where(data==el, data).mask)
    if not minval==None:
        ix = np.logical_or(ix, np.ma.masked_less(data, minval).mask)
    if not maxval==None:
        ix = np.logical_or(ix, np.ma.masked_greater(data, maxval).mask)

    return np.where(np.logical_not(ix))[0]


def meshgridN(*arrs):
    """N-dimensional meshgrid

    Just pass sequences of coordinates arrays

    """
    arrs = tuple(arrs)
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
##   return tuple(ans[::-1])
    return tuple(ans)


def gridaspoints(*arrs):
    """Creates an N-dimensional grid form arrs and returns grid points sequence of point coordinate pairs
    """
    # there is a small gotcha here.
    # with the convention following the 2013-08-30 sprint in Potsdam it was
    # agreed upon that arrays should have shapes (...,z,y,x) similar to the
    # convention that polar data should be (...,time,scan,azimuth,range)
    #
    # Still coordinate tuples are given in the order (x,y,z) [and hopefully not
    # more dimensions]. Therefore np.meshgrid must be fed the axis coordinates
    # in shape order (z,y,x) and the result needs to be reversed in order
    # for everything to work out.
    grid = tuple([dim.ravel()
                  for dim in reversed(np.meshgrid(*arrs, indexing='ij'))])
    return np.vstack(grid).transpose()


def issequence(x):
    """Test whether x is a sequence of numbers
    """
    out = True
    try:
        # can we get a length on the object
        length = len(x)
    except:
        return(False)
    # is the object not a string?
    out = np.all( np.isreal(x) )
    return out


def trapezoid(data, x1, x2, x3, x4):
    """
    Applied the trapezoidal function described in Vulpiani et al, 2012 to determine
    the degree of membership in the non-meteorological target class.

    Parameters
    ----------
    data : array
        Array contaning the data
    x1 : float
        x-value of the first vertex of the trapezoid
    x2 : float
        x-value of the second vertex of the trapezoid
    x3 : float
        x-value of the third vertex of the trapezoid
    x4 : float
        x-value of the fourth vertex of the trapezoid

    Returns
    -------
    d : array
        Array of values describing degree of membership in nonmeteorological target class.

    """

    d = np.ones(np.shape(data))
    d[np.logical_or(data <= x1, data >= x4)] = 0
    d[np.logical_and(data >= x2, data <= x3)] = 1
    d[np.logical_and(data > x1, data < x2)] = (data[np.logical_and(data > x1, data < x2)] - x1)/float((x2-x1))
    d[np.logical_and(data > x3, data < x4)] = (x4 - data[np.logical_and(data > x3, data < x4)])/float((x4-x3))

    d[np.isnan(data)] = np.nan

    return d



if __name__ == '__main__':
    print 'wradlib: Calling module <util> as main...'





