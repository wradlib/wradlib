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

"""
import numpy as np
import datetime as dt


def aggregate_in_time(src, dt_src, dt_trg, taxis=0, func='sum'):
    """Aggregate time series data to a coarser temporal resolution.

    Parameters
    ----------

    src : array of shape (..., original number of time steps,...)
        This is the time series data which should be aggregated. The position
        of the time dimension is indicated by the *taxis* argument. The number
        of time steps corresponds to the length pf the time dimension.

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

    func : numby function name, e.g. 'sum', 'mean'
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
    >>> print aggregate_in_time(src, dt_src, dt_trg)


    """
    src, dt_src, dt_trg = np.array(src), np.array(dt_src), np.array(dt_trg)
    trg_shape = list(src.shape)
    trg_shape[taxis] = len(dt_trg)-1
    trg = np.repeat(np.nan, _shape2size(trg_shape)).reshape(trg_shape)
    for i in range(len(dt_trg)-1):
        trg_slice = [slice(0,j) for j in trg.shape]
        trg_slice[taxis] = i
        src_slice = [slice(0,src.shape[j]) for j in range(src.ndim)]
        src_slice[taxis] = np.where( np.logical_and(dt_src<=dt_trg[i+1], dt_src>=dt_trg[i]) )[0][:-1]
        if len(src_slice[taxis])==0:
            trg[trg_slice] = np.nan
        else:
            trg[trg_slice] = _get_func(func)(src[src_slice], axis=taxis)
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


if __name__ == '__main__':
    print 'wradlib: Calling module <util> as main...'
