#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not
attributable to the other modules

.. autosummary::
   :nosignatures:
   :toctree: generated/

   aggregate_in_time
   aggregate_equidistant_tseries
   from_to
   filter_window_polar
   filter_window_cartesian
   find_bbox_indices
   get_raster_origin
   calculate_polynomial

"""
import datetime as dt
from datetime import tzinfo, timedelta
from time import mktime
import warnings
import functools
import os

import numpy as np
from scipy import interpolate
from scipy.ndimage import filters
from scipy.spatial import cKDTree
from osgeo import ogr

warnings.simplefilter('always', DeprecationWarning)
# warnings.simplefilter('always', FutureWarning)


def deprecated(replacement=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.

    Author: Giampaolo Rodola' <g.rodola [AT] gmail [DOT] com>
    License: MIT

    Parameters
    ----------
    replacement: string
        function name of replacement function

    >>> # Todo: warnings are sent to stderr instead of stdout
    >>> # so they are not seen here
    >>> from wradlib.util import deprecated
    >>> import sys
    >>> sys.stderr = sys.stdout  # noqa
    >>> @deprecated()
    ... def foo(x):
    ...     return x
    >>> ret = foo(1) #doctest: +ELLIPSIS
    /.../util.py:1: DeprecationWarning: wradlib.util.foo is deprecated
      #!/usr/bin/env python

    >>> def newfun(x):
    ...     return 0
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    >>> ret = foo(1) #doctest: +ELLIPSIS
    /.../util.py:1: DeprecationWarning: wradlib.util.foo is deprecated; \
use <function newfun at 0x...> instead
      #!/usr/bin/env python

    """

    def outer(fun):
        msg = "%s.%s is deprecated" % (fun.__module__, fun.__name__)
        if replacement is not None:
            msg += "; use %s instead" % replacement

        @functools.wraps(fun)
        def inner(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return fun(*args, **kwargs)

        return inner

    return outer


def apichange_kwarg(ver, par, typ, expar=None, exfunc=None, msg=None):
    """A decorator to generate a DeprectationWarning.

    .. versionadded:: 0.4.1

    This decorator function generates a DeprecationWarning if a given kwarg
    is changed/deprecated in a future version.

    The warning is only issued, when the kwarg is actually used in the
    function call.

    Parameters
    ----------

    ver : string
        Version from when the changes take effect
    par : string
        Name of parameter which is affected
    typ : Python type
        Data type of parameter which is affected
    expar : string
        Name of parameter to be used in future
    exfunc : function
        Function which can convert from old to new parameter
    msg : string
        additional warning message

    """

    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            para = kwargs.pop(par, None)
            key = par
            if para:
                wmsg = "\nPrevious behaviour of parameter '%s' in " \
                       "function '%s.%s' is deprecated " \
                       "\nand will be changed in version '%s'." % \
                       (par, func.__module__, func.__name__, ver)
                if expar:
                    wmsg += "\nUse parameter %s instead." % expar
                if exfunc:
                    wmsg += "\nWrong parameter types will be " \
                            "automatically converted by using %s.%s." % \
                            (exfunc.__module__, exfunc.__name__)
                if msg:
                    wmsg += "\n%s" % msg
                if isinstance(para, typ):
                    if exfunc:
                        para = exfunc(para)
                    if expar:
                        key = expar
                    warnings.warn(wmsg, category=DeprecationWarning,
                                  stacklevel=2)
                kwargs.update({key: para})
            return func(*args, **kwargs)

        return inner

    return outer


class OptionalModuleStub(object):
    """Stub class for optional imports.

    Objects of this class are instantiated when optional modules are not
    present on the user's machine.
    This allows global imports of optional modules with the code only breaking
    when actual attributes from this module are called.
    """

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        link = 'https://wradlib.github.io/wradlib-docs/latest/' \
               'gettingstarted.html#optional-dependencies'
        raise AttributeError('Module "{0}" is not installed.\n\n'
                             'You tried to access function/module/attribute '
                             '"{1}"\nfrom module "{0}".\nThis module is '
                             'optional right now in wradlib.\nYou need to '
                             'separately install this dependency.\n'
                             'Please refer to {2}\nfor further instructions.'.
                             format(self.name, name, link))


def import_optional(module):
    """Allowing for lazy loading of optional wradlib modules or dependencies.

    This function removes the need to satisfy all dependencies of wradlib
    before being able to work with it.

    Parameters
    ----------
    module : string
             name of the module

    Returns
    -------
    mod : object
          if module is present, returns the module object, on ImportError
          returns an instance of `OptionalModuleStub` which will raise an
          AttributeError as soon as any attribute is accessed.

    Examples
    --------
    Trying to import a module that exists makes the module available as normal.
    You can even use an alias. You cannot use the '*' notation, or import only
    select functions, but you can simulate most of the standard import syntax
    behavior
    >>> m = import_optional('math')
    >>> m.log10(100)
    2.0

    Trying to import a module that does not exists, does not produce
    any errors. Only when some function is used, the code triggers an error
    >>> m = import_optional('nonexistentmodule')  # noqa
    >>> m.log10(100)  # doctest +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: Module "nonexistentmodule" is not installed.
    <BLANKLINE>
    You tried to access function/module/attribute "log10"
    from module "nonexistentmodule".
    This module is optional right now in wradlib.
    You need to separately install this dependency.
    Please refer to https://wradlib.github.io/wradlib-docs/\
latest/gettingstarted.html#optional-dependencies
    for further instructions.
    """
    try:
        mod = __import__(module)
    except ImportError:
        mod = OptionalModuleStub(module)

    return mod


def aggregate_equidistant_tseries(tstart, tend, tdelta, tends_src, tdelta_src,
                                  src, method="sum", minpercvalid=100.):
    """Aggregates an equidistant time series to equidistant target time windows.

    This function aggregates time series data under the assumption that the
    source and the target time series are equidistant, and that the source time
    steps regularly fit into the target time steps (no shifts at the
    boundaries). This is the most trivial aggregation scenario.

    However, the function allows for gaps in the source data. This means, we
    assume the data to be equidistant (each item represents a time step with
    the same length), but it does not need to be contiguous. NaN values in the
    source data are allowed, too.

    The output, though, will be a contiguous time series. This series will have
    NaN values for those target time steps which were not sufficiently
    supported by source data. The decision whether the support was "sufficient"
    is based on the argument *minpercvalid*. This argument specifies the
    minimum percentage of valid source time steps inside one target time step
    (valid meaning not NaN and not missing) which is required to compute an
    aggregate. The default value of minpercvalid is 100 percent. This means no
    gaps are allowed, and a target time step will be NaN if only one source
    time step is missing.

    Aggregation methods at the moment are "sum" and "mean" of the source data.

    Parameters
    ----------
    tstart : isostring or datetime object
        start of the target time series
    tend : isostring or datetime object
        end of the target time series
    tdelta : integer
        resolution of the target time series (seconds)
    tends_src : sequence of isostrings or datetime objects
        timestamps which define the END of the source time steps
    tdelta_src : integer
        resolution of the source data (seconds)
    src : :class:`numpy:numpy.ndarray`
        1-d array of floats source values
    method : string
        Method of aggregation (either "sum" or "mean")
    minpercvalid : float
        Minimum percentage of valid source values within target interval that
        are required to compute an aggregate target value. If set to 100
        percent, the target value wil be NaN if only one source value is
        missing or NaN. If set to 90 percent, target value will be NaN if more
        than 10 percent of the source values are missing or NaN.

    Returns
    -------
    tstarts : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        array of timestamps which defines the start of each target time
        step/window
    tends : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        array of timestamps which defines the end of each target time
        step/window
    agg : :class:`numpy:numpy.ndarray`
        Array of aggregated values
        aggregated values for each target time step

    Examples
    --------
    >>> tstart = "2000-01-01 00:00:00"  # noqa
    >>> tend = "2000-01-02 00:00:00"
    >>> tdelta = 3600 * 6
    >>> tends_src = ["2000-01-01 02:00:00", "2000-01-01 03:00:00", \
    "2000-01-01 04:00:00", "2000-01-01 05:00:00", "2000-01-01 12:00:00"]
    >>> tdelta_src = 3600
    >>> src = [1, 1, 1, 1, 1]
    >>> tstarts, tends, agg = aggregate_equidistant_tseries(tstart, tend, \
    tdelta, tends_src, tdelta_src, src, minpercvalid=50.)
    >>> print(agg)
    [  4.  nan  nan  nan]

    """
    # Check arguments and make sure they have the right type
    src = np.array(src)
    tstart = iso2datetime(tstart)
    tend = iso2datetime(tend)
    tends_src = np.array([iso2datetime(item) for item in tends_src])
    twins = np.array(from_to(tstart, tend, tdelta))
    tstarts = twins[:-1]
    tends = twins[1:]

    # Check consistency of timestamps and data
    assert len(tends_src) == len(src), \
        "Length of source timestamps tends_src must " \
        "equal length of source data src."

    # Check that source time steps are sorted correctly
    assert np.all(np.sort(tends_src) == tends_src), \
        "The source time steps are not in chronological order."

    # number of expected source time steps per target timestep
    assert tdelta % tdelta_src == 0, \
        "Target resolution %d is not a multiple of " \
        "source resolution %d." % (tdelta, tdelta_src)
    nexpected = tdelta / tdelta_src

    # results container
    agg = np.repeat(np.nan, len(tstarts))

    inconsistencies = 0
    # iterate over target time windows
    for i, begin in enumerate(tstarts):
        end = tends[i]
        # select source data that is between start and end of this interval
        ixinside = np.where((tends_src > begin) & (tends_src <= end))[0]
        # Time steps and array shape in that interval sized as if
        # it had no gaps
        tends_src_expected = np.array(from_to(begin, end, tdelta_src)[1:])
        srcfull = np.repeat(np.nan, len(tends_src_expected))
        # These are the indices of srcfull which actually have
        # data according to src
        validix = np.where(np.in1d(tends_src_expected, tends_src[ixinside]))[0]
        if not len(validix) == len(ixinside):
            # If we find time stamps in tends_src[ixinside] that are not
            #   contained in the expected cintiguous time steps
            #   (tends_src_expected),
            #   we assume that the data is corrupt (can have multiple reasons,
            #   e.g. wrong increment)
            inconsistencies += 1
            continue
        srcfull[validix] = src[ixinside]
        # valid source values found in target time window
        nvalid = len(np.where(~np.isnan(srcfull))[0])
        if nvalid > nexpected:
            # If we have more source time steps in target interval than
            # expected aomething must be wrong.
            inconsistencies += 1
            continue
        if float(nvalid) / float(nexpected) >= minpercvalid / 100.:
            if method == "sum":
                agg[i] = np.nansum(srcfull)
            elif method == "mean":
                agg[i] = np.nanmean(srcfull)
            else:
                print("Aggregation method not known, yet.")
                raise Exception()
    if inconsistencies > 0:
        print("WARNING: Inconsistent source times "
              "in %d target time intervals." % inconsistencies)

    return tstarts, tends, agg


def aggregate_in_time(src, dt_src, dt_trg, taxis=0, func='sum'):
    """Aggregate time series data to a coarser temporal resolution.

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The position
        of the time dimension is indicated by the *taxis* argument. The number
        of time steps corresponds to the length of the time dimension.
    taxis : integer
        This is the position of the time dimension in array *src*.
    dt_src : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is
        the upper limit of time step 1 and the lower limit of time step 2 and
        so on.
    dt_trg : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *number of output time steps + 1* analogously to
        dt_src. This means: dt_trg[0] is the lower limit of output time step 1,
        dt_trg[1] is the upper limit of output time step 1 and the lower limit
        of output time step 2 and so on.
    func : str
        numpy function name, e.g. 'sum', 'mean'
        Defines the way the data should be aggregated. The string must
        correspond to a valid numpy function, e.g. 'sum', 'mean', 'min', 'max'.

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the
        array *dt_trg* which defines the limits of the output time step
        intervals.

    Examples
    --------
    >>> src = np.arange(8 * 4).reshape((8, 4))
    >>> print('source time series:') # doctest: +SKIP
    >>> print(src)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]
     [24 25 26 27]
     [28 29 30 31]]
    >>> dt_src = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + \
    dt.timedelta(hours=i) for i in range(9)]
    >>> print('source time interval limits:') # doctest: +SKIP
    >>> for tim in dt_src: print(tim)
    2008-06-02 00:00:00
    2008-06-02 01:00:00
    2008-06-02 02:00:00
    2008-06-02 03:00:00
    2008-06-02 04:00:00
    2008-06-02 05:00:00
    2008-06-02 06:00:00
    2008-06-02 07:00:00
    2008-06-02 08:00:00
    >>> print('target time interval limits:') # doctest: +SKIP
    >>> dt_trg = [dt.datetime.strptime('2008-06-02', '%Y-%m-%d' ) + \
    dt.timedelta(seconds=i*3600*4) for i in range(4)]
    >>> for tim in dt_trg: print(tim)
    2008-06-02 00:00:00
    2008-06-02 04:00:00
    2008-06-02 08:00:00
    2008-06-02 12:00:00
    >>> print('target time series') # doctest: +SKIP
    >>> print(aggregate_in_time(src, dt_src, dt_trg, taxis=0, func='sum'))
    [[  24.   28.   32.   36.]
     [  88.   92.   96.  100.]
     [  nan   nan   nan   nan]]

    See :ref:`notebooks/basics/wradlib_workflow.ipynb#Rainfall-accumulation`.

    """
    # src, dt_src, dt_trg = np.array(src), np.array(dt_src), np.array(dt_trg)
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)
    trg_shape = list(src.shape)
    trg_shape[taxis] = len(dt_trg) - 1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)
    for i in range(len(dt_trg) - 1):
        trg_slice = [slice(0, j) for j in trg.shape]
        trg_slice[taxis] = i
        src_slice = [slice(0, src.shape[j]) for j in range(len(src.shape))]
        src_slice[taxis] = np.where(
            np.logical_and(dt_src <= dt_trg[i + 1],
                           dt_src >= dt_trg[i]))[0][:-1]
        if len(src_slice[taxis]) == 0:
            trg[trg_slice] = np.nan
        else:
            trg[trg_slice] = _get_func(func)(src[tuple(src_slice)], axis=taxis)
    return trg


def sum_over_time_windows(src, dt_src, dt_trg, minpercvalid):
    """Returns the sums of time series <src> within the time windows dt_trg

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The number
        of time steps corresponds to the length of the time dimension.
    dt_src : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is
        the upper limit of time step 1 and the lower limit of time step 2 and
        so on.
    dt_trg : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *number of output time steps + 1* analogously to
        dt_src. This means: dt_trg[0] is the lower limit of output time step 1,
        dt_trg[1] is the upper limit of output time step 1 and the lower limit
        of output time step 2 and so on.
    # Todo: add minpercvalid
    """
    assert len(src) + 1 == len(
        dt_src), "Length of time series array <src> must be one " \
                 "less than datetime array <dt_src>."
    try:
        dt_src = [dt.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S")
                  for dtime in dt_src]
    except TypeError:
        pass
    try:
        dt_trg = [dt.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S")
                  for dtime in dt_trg]
    except TypeError:
        pass

    accum = np.repeat(np.nan, len(dt_trg) - 1)

    for i, tstart in enumerate(dt_trg[:-1]):
        tend = dt_trg[i + 1]
        # accumulate gage data to target time windows
        ix = np.where((dt_src > tstart) & (dt_src <= tend))[0] - 1
        if len(src[ix]) == 0:
            continue
        elif (len(np.where(np.isnan(src[ix]))[0]) / len(src[ix])) < \
                (minpercvalid / 100.):
            accum[i] = np.nansum(src[ix])
    return accum


def mean_over_time_windows(src, dt_src, dt_trg, minbasepoints=1):
    """UNDER DEVELOPMENT: Aggregate time series data to a coarser temporal
    resolution.

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The number
        of time steps corresponds to the length of the time dimension.
    dt_src : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is
        the upper limit of time step 1 and the lower limit of time step 2 and
        so on.
    dt_trg : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *number of output time steps + 1* analogously to
        dt_src. This means: dt_trg[0] is the lower limit of output time step 1,
        dt_trg[1] is the upper limit of output time step 1 and the lower limit
        of output time step 2 and so on.
    #todo: add minbasepoints

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the
        array *dt_trg* which defines the limits of the output time step
        intervals.

    Examples
    --------
    >>> # TODO: put an example here for `mean_over_time_windows`

    """
    # Convert input time steps to numpy arrays
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)
    # Create a new container for the target data
    trg_shape = list(src.shape)
    trg_shape[0] = len(dt_trg) - 1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)
    for i in range(len(dt_trg) - 1):
        # width of window
        width = float(_tdelta2seconds(dt_trg[i + 1] - dt_trg[i]))
        # These are the intervals completely INSIDE the target time window
        src_ix = np.where(np.logical_and(dt_src > dt_trg[i],
                                         dt_src < dt_trg[i + 1]))[0]
        intervals = dt_src[src_ix[1:]] - dt_src[src_ix[:-1]]
        # check left edge
        intervals = np.insert(intervals, 0, dt_src[src_ix[0]] - dt_trg[i])
        if src_ix[0] > 0:
            src_ix = np.insert(src_ix, 0, src_ix[0] - 1)
        # check right edge
        intervals = np.append(intervals, dt_trg[i + 1] - dt_src[src_ix[-1]])
        if src_ix[-1] > (len(dt_src) - 1):
            src_ix = np.append(src_ix, src_ix[-1] + 1)
        # convert to seconds
        intervals = np.array([_tdelta2seconds(interval)
                              for interval in intervals])
        # compute weights
        weights = intervals / width
        # compute weighted mean
        trg[i] = np.dot(np.transpose(src[src_ix]), weights)
    return trg


def average_over_time_windows(src, dt_src, dt_trg, maxdist=3600,
                              helper_interval=300, **ipargs):
    """UNDER DEVELOPMENT: Computes the average of a time series over given
    time windows.

    This function computes the average values of an irregular time series
    ``src`` within given time windows ``dt_trg``. The datetimes of the original
    time series are given by ``dt_src``. The idea of this function is to create
    regular helper timesteps at an interval length given by
    ``helper_interval``. The values of ``src`` are then interpolated to these
    helper time steps, and the resulting helper values are finally averaged
    over the given target time windows.


    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The number
        of time steps corresponds to the length of the time dimension.
    dt_src : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *original number of time steps + 1* because dt_src
        defines the limits of the intervals corresponding to the time steps.
        This means: dt_src[0] is the lower limit of time step 1, dt_src[1] is
        the upper limit of time step 1 and the lower limit of time step 2 and
        so on.
    dt_trg : :class:`numpy:numpy.ndarray`
        Array of datetime objects
        Must be of length *number of output time steps + 1* analogously to
        dt_src. This means: dt_trg[0] is the lower limit of output time step 1,
        dt_trg[1] is the upper limit of output time step 1 and the lower limit
        of output time step 2 and so on.
    # todo: add maxdist, helper_interval, **ipargs

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array of shape (..., len(dt_trg) - 1, ...)
        The length of the time dimension of the output array depends on the
        array *dt_trg* which defines the limits of the output time step
        intervals.

    Examples
    --------
    >>> # TODO: put an example here for `average_over_time_windows`

    """
    # Convert input time steps to numpy arrays
    dt_src, dt_trg = np.array(dt_src), np.array(dt_trg)

    trg_secs = np.array([mktime(tstep.timetuple()) for tstep in dt_trg])
    src_secs = np.array([mktime(tstep.timetuple()) for tstep in dt_src])
    helper_secs = np.arange(trg_secs[0], trg_secs[-1], helper_interval)

    # Interpolate to target points
    f = interpolate.interp1d(src_secs, src, axis=0, bounds_error=False)
    helpers = f(helper_secs)

    # Mask those values as invalid which are more than maxdist from the next
    # source point
    tree = cKDTree(src_secs.reshape((-1, 1)))
    dists, ix = tree.query(helper_secs.reshape((-1, 1)), k=1)
    # deal with edges (in case of extrapolation, we apply nearest neighbour)
    np.where(np.isnan(helpers), src[ix], helpers)
    # mask out points which are to far from the next source point
    helpers[np.where(dists > maxdist)[0]] = np.nan

    # Create a new container for the target data
    trg_shape = list(src.shape)
    trg_shape[0] = len(dt_trg) - 1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)

    for i in range(len(dt_trg) - 1):
        # width of window
        # width = float(_tdelta2seconds(dt_trg[i + 1] - dt_trg[i]))
        # These are the intervals completely INSIDE the target time window
        helper_ix = np.where(np.logical_and(dt_src >= dt_trg[i],
                                            dt_src <= dt_trg[i + 1]))[0]
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
        func = getattr(np, funcname)
    except AttributeError:
        raise AttributeError('<' + funcname +
                             '> is not a valid function in numpy...')
    return func


def _shape2size(shape):
    """
    Compute the size which corresponds to a shape
    """
    out = 1
    for item in shape:
        out *= item
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
    if not type(tstart) == dt.datetime:
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not type(tend) == dt.datetime:
        tend = dt.datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    tdelta = dt.timedelta(seconds=tdelta)
    tsteps = [tstart, ]
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
    if not isinstance(tstart, dt.datetime):
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not isinstance(tend, dt.datetime):
        tend = dt.datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    if not as_secs:
        return tend - tstart
    else:
        return _tdelta2seconds(tend - tstart)


def iso2datetime(iso):
    """Converts an ISO formatted time string to a datetime object.

    Parameters
    ----------
    iso : string
        time string

    """
    # in case the argument has been parsed to datetime before
    if type(iso) == dt.datetime:
        return iso
    # sometimes isoformat separates date and time by a white space
    iso = iso.replace(" ", "T")
    try:
        return dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%f")
    except (ValueError, TypeError):
        return dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        print("Could not convert argument <%r> to datetime. "
              "Probably not an isostring. See following traceback:" % iso)
        raise


def timestamp2index(ts, delta, refts, **kwargs):
    """Calculates the array index for a certain time in an equidistant
    time-series given the reference time (where the index would be 0)
    and the time discretization.
    If any of the input parameters contains timezone information, all others
    also need to contain timezone information.

    Parameters
    ----------
    ts : str or datetime-object
        The timestamp to determine the index for.
        If it is a string, it will be converted to datetime using the
        function iso2datetime
    delta : str or timedelta object
        The discretization of the time series (the amount of time that
        elapsed between indices)
        If used as a string, it needs to be given in the format
        "keyword1=value1,keyword2=value2". Keywords must be understood
        by the timedelta constructor (like days, hours,
        minutes, seconds) and the values may only be integers.
    refts : str or datetime-object
        The timestamp to determine the index for
        If it is a string, it will be converted to datetime using the
        function iso2datetime.

    Returns
    -------
    index : integer
        The index of a discrete time series array of the given parameters.

    Example
    -------
    >>> import datetime as dt
    >>> timestr1, timestr2 = '2008-06-01T00:00:00', '2007-01-01T00:00:00'
    >>> timestamp2index(timestr1, 'minutes=5', timestr2)
    148896
    >>> timestamp2index(timestr1, 'hours=1,minutes=5',timestr2)
    11453
    >>> timestamp2index(timestr1, dt.timedelta(hours=1, minutes=5), timestr2)
    11453
    """
    if not isinstance(ts, dt.datetime):
        _ts = iso2datetime(ts)
    else:
        _ts = ts
    if not isinstance(refts, dt.datetime):
        _refts = iso2datetime(refts)
    else:
        _refts = refts
    if not isinstance(delta, dt.timedelta):
        kwargs = dict([(sp[0], int(sp[1]))
                       for sp in [item.split('=')
                                  for item in delta.split(',')]])
        _dt = dt.timedelta(**kwargs)
    else:
        _dt = delta
    return int(_tdelta2seconds(_ts - _refts) / _tdelta2seconds(_dt))


def _idvalid(data, isinvalid=None, minval=None, maxval=None):
    """Identifies valid entries in an array and returns the corresponding
    indices

    Invalid values are NaN and Inf. Other invalid values can be passed using
    the isinvalid keyword argument.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray` of floats
    isinvalid : list of what is considered an invalid value

    """
    if isinvalid is None:
        isinvalid = [-99., 99, -9999., -9999]
    ix = np.ma.masked_invalid(data).mask
    for el in isinvalid:
        ix = np.logical_or(ix, np.ma.masked_where(data == el, data).mask)
    if minval is not None:
        ix = np.logical_or(ix, np.ma.masked_less(data, minval).mask)
    if maxval is not None:
        ix = np.logical_or(ix, np.ma.masked_greater(data, maxval).mask)

    return np.where(np.logical_not(ix))[0]


def meshgridN(*arrs):
    """N-dimensional meshgrid

    Just pass sequences of coordinates arrays

    """
    arrs = tuple(arrs)
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    # return tuple(ans[::-1])
    return tuple(ans)


def gridaspoints(*arrs):
    """Creates an N-dimensional grid form arrs and returns grid points sequence
    of point coordinate pairs
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

    Parameters
    ----------
    x : sequence to test

    """
    out = True
    try:
        # can we get a length on the object
        len(x)
    except Exception:
        return False
    # is the object not a string?
    out = np.all(np.isreal(x))
    return out


def trapezoid(data, x1, x2, x3, x4):
    """
    Applied the trapezoidal function described in :cite:`Vulpiani`
    to determine the degree of membership in the non-meteorological
    target class.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array containing the data
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
    d : :class:`numpy:numpy.ndarray`
        Array of values describing degree of membership in
        nonmeteorological target class.

    """

    d = np.ones(np.shape(data))
    d[np.logical_or(data <= x1, data >= x4)] = 0
    d[np.logical_and(data >= x2, data <= x3)] = 1
    d[np.logical_and(data > x1, data < x2)] = \
        (data[np.logical_and(data > x1, data < x2)] - x1) / float((x2 - x1))
    d[np.logical_and(data > x3, data < x4)] = \
        (x4 - data[np.logical_and(data > x3, data < x4)]) / float((x4 - x3))

    d[np.isnan(data)] = np.nan

    return d


def maximum_intensity_projection(data, r=None, az=None, angle=None,
                                 elev=None, autoext=True):
    """
    Computes the maximum intensity projection along an arbitrary cut
    through the ppi from polar data.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array containing polar data (azimuth, range)
    r : :class:`numpy:numpy.ndarray`
        Array containing range data
    az : array
        Array containing azimuth data
    angle : float
        angle of slice, Defaults to 0. Should be between 0 and 180.
        0. means horizontal slice, 90. means vertical slice
    elev : float
        elevation angle of scan, Defaults to 0.
    autoext : True | False
        This routine uses numpy.digitize to bin the data.
        As this function needs bounds, we create one set of coordinates more
        than would usually be provided by `r` and `az`.

    Returns
    -------
    xs : :class:`numpy:numpy.ndarray`
        meshgrid x array
    ys : :class:`numpy:numpy.ndarray`
        meshgrid y array
    mip : :class:`numpy:numpy.ndarray`
        Array containing the maximum intensity projection (range, range*2)

    """

    from wradlib.georef import beam_height_n as beam_height_n

    # this may seem odd at first, but d1 and d2 are also used in several
    # plotting functions and thus it may be easier to compare the functions
    d1 = r
    d2 = az

    # providing 'reasonable defaults', based on the data's shape
    if d1 is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    if d2 is None:
        d2 = np.arange(data.shape[0], dtype=np.float)

    if angle is None:
        angle = 0.0

    if elev is None:
        elev = 0.0

    if autoext:
        # the ranges need to go 'one bin further', assuming some regularity
        # we extend by the distance between the preceding bins.
        x = np.append(d1, d1[-1] + (d1[-1] - d1[-2]))
        # the angular dimension is supposed to be cyclic, so we just add the
        # first element
        y = np.append(d2, d2[0])
    else:
        # no autoext basically is only useful, if the user supplied the correct
        # dimensions himself.
        x = d1
        y = d2

    # roll data array to specified azimuth, assuming equidistant azimuth angles
    ind = (d2 >= angle).nonzero()[0][0]
    data = np.roll(data, ind, axis=0)

    # build cartesian range array, add delta to last element to compensate for
    # open bound (np.digitize)
    dc = np.linspace(-np.max(d1), np.max(d1) + 0.0001, num=d1.shape[0] * 2 + 1)

    # get height values from polar data and build cartesian height array
    # add delta to last element to compensate for open bound (np.digitize)
    hp = np.zeros((y.shape[0], x.shape[0]))
    hc = beam_height_n(x, elev)
    hp[:] = hc
    hc[-1] += 0.0001

    # create meshgrid for polar data
    xx, yy = np.meshgrid(x, y)

    # create meshgrid for cartesian slices
    xs, ys = np.meshgrid(dc, hc)
    # xs, ys = np.meshgrid(dc,x)

    # convert polar coordinates to cartesian
    xxx = xx * np.cos(np.radians(90. - yy))
    # yyy = xx * np.sin(np.radians(90.-yy))

    # digitize coordinates according to cartesian range array
    range_dig1 = np.digitize(xxx.ravel(), dc)
    range_dig1.shape = xxx.shape

    # digitize heights according polar height array
    height_dig1 = np.digitize(hp.ravel(), hc)
    # reshape accordingly
    height_dig1.shape = hp.shape

    # what am I doing here?!
    range_dig1 = range_dig1[0:-1, 0:-1]
    height_dig1 = height_dig1[0:-1, 0:-1]

    # create height and range masks
    height_mask = [(height_dig1 == i).ravel().nonzero()[0]
                   for i in range(1, len(hc))]
    range_mask = [(range_dig1 == i).ravel().nonzero()[0]
                  for i in range(1, len(dc))]

    # create mip output array, set outval to inf
    mip = np.zeros((d1.shape[0], 2 * d1.shape[0]))
    mip[:] = np.inf

    # fill mip array,
    # in some cases there are no values found in the specified range and height
    # then we fill in nans and interpolate afterwards
    for i in range(0, len(range_mask)):
        mask1 = range_mask[i]
        found = False
        for j in range(0, len(height_mask)):
            mask2 = np.intersect1d(mask1, height_mask[j])
            # this is to catch the ValueError from the max() routine when
            # calculating on empty array
            try:
                mip[j, i] = data.ravel()[mask2].max()
                if not found:
                    found = True
            except ValueError:
                if found:
                    mip[j, i] = np.nan

    # interpolate nans inside image, do not touch outvals
    good = ~np.isnan(mip)
    xp = good.ravel().nonzero()[0]
    fp = mip[~np.isnan(mip)]
    x = np.isnan(mip).ravel().nonzero()[0]
    mip[np.isnan(mip)] = np.interp(x, xp, fp)

    # reset outval to nan
    mip[mip == np.inf] = np.nan

    return xs, ys, mip


def filter_window_polar(img, wsize, fun, rscale, random=False):
    r"""Apply a filter of an approximated square window of half size `fsize`
    on a given polar image `img`.

    Parameters
    ----------
    img : :class:`numpy:numpy.ndarray`
        2d array of values to which the filter is to be applied
    wsize : float
        Half size of the window centred on the pixel [m]
    fun : string
        name of the 1d filter from :mod:`scipy:scipy.ndimage`
    rscale : float
        range [m] scale of the polar grid
    random: bool
        True to use random azimuthal size to avoid long-term biases.

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array with the same shape as `img`, containing the filter's results.

    """
    ascale = 2 * np.pi / img.shape[0]
    data_filtered = np.empty(img.shape, dtype=img.dtype)
    fun = getattr(filters, "%s_filter1d" % fun)
    nbins = img.shape[-1]
    ranges = np.arange(nbins) * rscale + rscale / 2
    asize = ranges * ascale
    if random:
        na = prob_round(wsize / asize).astype(int)
    else:
        na = np.fix(wsize / asize + 0.5).astype(int)
    # Maximum of adjacent azimuths (higher close to the origin) to
    # increase performance
    na[na > 20] = 20
    sr = np.fix(wsize / rscale + 0.5).astype(int)
    for sa in np.unique(na):
        imax = np.where(na >= sa)[0][-1] + 1
        imin = np.where(na <= sa)[0][0]
        if sa == 0:
            data_filtered[:, imin:imax] = img[:, imin:imax]
        imin2 = max(imin - sr, 0)
        imax2 = min(imax + sr, nbins)
        temp = img[:, imin2:imax2]
        temp = fun(temp, size=2 * sa + 1, mode='wrap', axis=0)
        temp = fun(temp, size=2 * sr + 1, axis=1)
        imin3 = imin - imin2
        imax3 = imin3 + imax - imin
        data_filtered[:, imin:imax] = temp[:, imin3:imax3]
    return data_filtered


def prob_round(x, prec=0):
    """Round the float number `x` to the lower or higher integer randomly
    following a binomial distribution

    Parameters
    ----------
    x : float
    prec : precision
    """
    fixup = np.sign(x) * 10 ** prec
    x *= fixup
    intx = x.astype(int)
    round_func = intx + np.random.binomial(1, x - intx)
    return round_func / fixup


def filter_window_cartesian(img, wsize, fun, scale, **kwargs):
    r"""Apply a filter of square window size `fsize` on a given
    cartesian image `img`.

    Parameters
    ----------
    img : :class:`numpy:numpy.ndarray`
        2d array of values to which the filter is to be applied
    wsize : float
        Half size of the window centred on the pixel [m]
    fun : string
        name of the 2d filter from :mod:`scipy:scipy.ndimage`
    scale : tuple of 2 floats
        x and y scale of the cartesian grid [m]

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array with the same shape as `img`, containing the filter's results.

    """
    fun = getattr(filters, "%s_filter" % fun)
    size = np.fix(wsize / scale + 0.5).astype(int)
    data_filtered = fun(img, size, **kwargs)
    return data_filtered


def roll2d_polar(img, shift=1, axis=0):
    r"""Roll a 2D polar array [azimuth,range] by a given `shift` for
    the given `axis`

    Parameters
    ----------
    img : :class:`numpy:numpy.ndarray`
        2d data array
    shift : int
        shift to apply to the array
    axis : int
        axis which will be shifted
    Returns
    -------
    out: new array with shifted values
    """
    if shift == 0:
        return img
    else:
        out = np.empty(img.shape)
    n = img.shape[axis]
    if axis == 0:
        if shift > 0:
            out[shift:, :] = img[:-shift, :]
            out[:shift, :] = img[n - shift:, :]
        else:
            out[:shift, :] = img[-shift:, :]
            out[n + shift:, :] = img[:-shift:, :]
    else:
        if shift > 0:
            out[:, shift:] = img[:, :-shift]
            out[:, :shift] = np.nan
        else:
            out[:, :shift] = img[:, -shift:]
            out[:, n + shift:] = np.nan
    return out


class UTC(tzinfo):
    """
    UTC implementation for tzinfo.

    See e.g. http://python.active-venture.com/lib/datetime-tzinfo.html

    Replaces pytz.utc
    """

    def __repr__(self):
        return "<UTC>"

    def utcoffset(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return timedelta(0)


def half_power_radius(r, bwhalf):
    """
    Half-power radius.

    .. versionadded:: 0.6.0

    ported from PyRadarMet

    Battan (1973),

    Parameters
    ----------
    r : float, :class:`numpy:numpy.ndarray` of floats
        Range from radar [m]
    bwhalf : float
        Half-power beam width [degrees]

    Returns
    -------
    Rhalf : float, :class:`numpy:numpy.ndarray` of floats
        Half-power radius [m]

    Examples
    --------
    Rhalf = half_power_radius(r,bwhalf)
    """

    Rhalf = (r * np.deg2rad(bwhalf)) / 2.

    return Rhalf


def get_raster_origin(coords):
    """

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        3 dimensional array (rows, cols, 2) of xy-coordinates

    Returns
    -------
    out : str
        'lower' or 'upper'

    """
    return 'lower' if (coords[1, 1] - coords[0, 0])[1] > 0 else 'upper'


def find_bbox_indices(coords, bbox):
    """
    Find min/max-indices for NxMx2 array coords using bbox-values.

    The bounding box is defined by two points (llx,lly and urx,ury)
    It finds the first indices before llx,lly and the first indices
    after urx,ury. If no index is found 0 and N/M is returned.

    .. versionadded:: 0.6.0

    .. versionchanged:: 0.10.0
       Find indices no matter if the coordinate origin is `lower` or `upper`.


    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        3 dimensional array (ny, nx, lon/lat) of floats
    bbox : 4-element :class:`numpy:numpy.ndarray`, list or tuple of floats
        (llx,lly,urx,ury)

    Returns
    -------
    bbind : tuple
        4-element tuple of int (llx,lly,urx,ury)
    """

    # sort arrays
    x_sort = np.argsort(coords[0, :, 0])
    y_sort = np.argsort(coords[:, 0, 1])

    # find indices in sorted arrays
    llx = np.searchsorted(coords[0, :, 0], bbox[0], side='left',
                          sorter=x_sort)
    urx = np.searchsorted(coords[0, :, 0], bbox[2], side='right',
                          sorter=x_sort)
    lly = np.searchsorted(coords[:, 0, 1], bbox[1], side='left',
                          sorter=y_sort)
    ury = np.searchsorted(coords[:, 0, 1], bbox[3], side='right',
                          sorter=y_sort)

    # get indices in original array
    if llx < len(x_sort):
        llx = x_sort[llx]
    if urx < len(x_sort):
        urx = x_sort[urx]
    if lly < len(y_sort):
        lly = y_sort[lly]
    if ury < len(y_sort):
        ury = y_sort[ury]

    # check at boundaries
    if llx:
        llx -= 1
    if get_raster_origin(coords) is 'lower':
        if lly:
            lly -= 1
    else:
        if lly < coords.shape[0]:
            lly += 1

    bbind = (llx, min(lly, ury), urx, max(lly, ury))

    return bbind


def has_geos():
    pnt1 = ogr.CreateGeometryFromWkt('POINT(10 20)')
    pnt2 = ogr.CreateGeometryFromWkt('POINT(30 20)')
    ogrex = ogr.GetUseExceptions()
    ogr.DontUseExceptions()
    hasgeos = pnt1.Union(pnt2) is not None
    if ogrex:
        ogr.UseExceptions()
    return hasgeos


def get_wradlib_data_path():
    wrl_data_path = os.environ.get('WRADLIB_DATA', None)
    if wrl_data_path is None:
        raise EnvironmentError("'WRADLIB_DATA' environment variable not set")
    if not os.path.isdir(wrl_data_path):
        raise EnvironmentError("'WRADLIB_DATA' path '{0}' "
                               "does not exist".format(wrl_data_path))
    return wrl_data_path


def get_wradlib_data_file(relfile):
    data_file = os.path.join(get_wradlib_data_path(), relfile)
    if not os.path.exists(data_file):
        raise EnvironmentError("WRADLIB_DATA file '{0}' "
                               "does not exist".format(data_file))
    return data_file


def calculate_polynomial(data, w):
    """Calculate Polynomial

    The functions calculates the following polynomial:

    .. math::

        P = \sum_{n=0}^{N} w(n) \cdot data^{n}

    .. versionadded:: 0.10.0

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Flat array of data values.
    w : :class:`numpy:numpy.ndarray`
        Array of shape (N) containing weights.

    Returns
    -------
    poly : :class:`numpy:numpy.ndarray`
        Flat array of processed data.
    """
    poly = np.zeros_like(data)
    for i, c in enumerate(w):
        poly += c * data**i
    return poly


if __name__ == '__main__':
    print('wradlib: Calling module <util> as main...')
