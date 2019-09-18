#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not
attributable to the other modules

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ['from_to', 'filter_window_polar', 'filter_window_cartesian',
           'find_bbox_indices', 'get_raster_origin', 'calculate_polynomial']
__doc__ = __doc__.format('\n   '.join(__all__))

import importlib
import datetime as dt
import os

import deprecation
import numpy as np
from osgeo import gdal, ogr
from scipy import ndimage, signal

from wradlib import version


@deprecation.deprecated(deprecated_in="1.6", removed_in="2.0",
                        current_version=version.version,
                        details="Use `wradlib.georef.maximum_intensity_"
                                "projection` instead.")
def maximum_intensity_projection(*args, **kwargs):
    from wradlib.georef import polar
    return polar.maximum_intensity_projection(*args, **kwargs)


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
    >>> m.log10(100)  #doctest: +ELLIPSIS
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
        mod = importlib.import_module(module)
    except ImportError:
        mod = OptionalModuleStub(module)

    return mod


def _shape_to_size(shape):
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


def meshgrid_n(*arrs):
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
    except TypeError:
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


def filter_window_polar(img, wsize, fun, rscale, random=False):
    """Apply a filter of an approximated square window of half size `fsize` \
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
    fun = getattr(ndimage.filters, "%s_filter1d" % fun)
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
    """Apply a filter of square window size `fsize` on a given \
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
    fun = getattr(ndimage.filters, "%s_filter" % fun)
    size = np.fix(wsize / scale + 0.5).astype(int)
    data_filtered = fun(img, size, **kwargs)
    return data_filtered


def roll2d_polar(img, shift=1, axis=0):
    """Roll a 2D polar array [azimuth,range] by a given `shift` for \
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


class UTC(dt.tzinfo):
    """UTC implementation for tzinfo.

    See e.g. http://python.active-venture.com/lib/datetime-tzinfo.html

    Replaces pytz.utc
    """

    def __repr__(self):
        return "<UTC>"

    def utcoffset(self, dtime):
        return dt.timedelta(0)

    def tzname(self, dtime):
        return "UTC"

    def dst(self, dtime):
        return dt.timedelta(0)


def half_power_radius(r, bwhalf):
    """
    Half-power radius.

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
    rhalf = half_power_radius(r,bwhalf)
    """

    rhalf = (r * np.deg2rad(bwhalf)) / 2.

    return rhalf


def get_raster_origin(coords):
    """Return raster origin

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
    """Find min/max-indices for NxMx2 array coords using bbox-values.

    The bounding box is defined by two points (llx,lly and urx,ury)
    It finds the first indices before llx,lly and the first indices
    after urx,ury. If no index is found 0 and N/M is returned.

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
    if get_raster_origin(coords) == 'lower':
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
    gdalex = gdal.GetUseExceptions()
    gdal.DontUseExceptions()
    ogr.DontUseExceptions()
    hasgeos = pnt1.Union(pnt2) is not None
    if ogrex:
        ogr.UseExceptions()
    if gdalex:
        gdal.UseExceptions()
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
    data_file = os.path.abspath(os.path.join(get_wradlib_data_path(), relfile))
    if not os.path.exists(data_file):
        raise EnvironmentError("WRADLIB_DATA file '{0}' "
                               "does not exist".format(data_file))
    return data_file


def calculate_polynomial(data, w):
    """Calculate Polynomial

    The functions calculates the following polynomial:

    .. math::

       P = \\sum_{n=0}^{N} w(n) \\cdot data^{n}

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


def medfilt_along_axis(x, n, axis=-1):
    """Applies median filter smoothing on one axis of an N-dimensional array.
    """
    kernel_size = np.array(x.shape)
    kernel_size[:] = 1
    kernel_size[axis] = n
    return signal.medfilt(x, kernel_size)


def gradient_along_axis(x):
    """Computes gradient along last axis of an N-dimensional array
    """
    axis = -1
    newshape = np.array(x.shape)
    newshape[axis] = 1
    diff_begin = (x[..., 1] - x[..., 0]).reshape(newshape)
    diff_end = (x[..., -1] - x[..., -2]).reshape(newshape)
    diffs = ((x - np.roll(x, 2, axis)) / 2.)
    diffs = np.append(diffs[..., 2:], diff_end, axis=axis)
    return np.insert(diffs, [0], diff_begin, axis=axis)


def gradient_from_smoothed(x, n=5):
    """Computes gradient of smoothed data along final axis of an array
    """
    return gradient_along_axis(medfilt_along_axis(x, n)).astype("f4")


def binned_statistic_dd(sample, values=None, binnumbers=None,
                        statistic='mean', bins=10, ranges=None,
                        expand_binnumbers=False):
    """
    Forked from scipy.stats.binned_statistic_dd

    Compute a multidimensional binned statistic for a set of data.

    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.

    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of D arrays of length N, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : sequence or int, optional
        The bin specification must be in one of the following forms:

          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... = bins).
          * The number of bins for all dimensions (nx = ny = ... = bins).

    ranges : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section of
        `binned_statistic_2d`.

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    binnumber : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.


    See Also
    --------
    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d

    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open in each dimension.  In
    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
    last bin, however, is ``[3, 4]``, which *includes* 4.

    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    If 'True': The returned `binnumber` is a shape (D,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.

    .. versionadded:: 0.11.0

    """
    known_stats = ['mean', 'median']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError('invalid statistic %r' % (statistic,))

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape

    nbin = np.empty(Ndim, int)    # Number of bins in each dimension
    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if ranges is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        smin = np.zeros(Ndim)
        smax = np.zeros(Ndim)
        for i in range(Ndim):
            smin[i], smax[i] = ranges[i]

    # Make sure the bins have a finite width.
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    if binnumbers is None:
        # Compute the bin number each sample falls into, in each dimension
        sampBin = [
            np.digitize(sample[:, i], edges[i])
            for i in range(Ndim)
        ]

        # Using `digitize`, values that fall on an edge are put
        # in the right bin.
        # For the rightmost bin, we want values equal to the right
        # edge to be counted in the last bin, and not as an outlier.
        for i in range(Ndim):
            # Find the rounding precision
            decimal = int(-np.log10(dedges[i].min())) + 6
            # Find which points are on the rightmost edge.
            on_edge = np.where(np.around(sample[:, i], decimal) ==
                               np.around(edges[i][-1], decimal))[0]
            # Shift these points one bin to the left.
            sampBin[i][on_edge] -= 1

        # Compute the sample indices in the flattened statistic matrix.
        binnumbers = np.ravel_multi_index(sampBin, nbin)

    if values is None:
        return(binnumbers)

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if(statistic != 'count' and Vlen != Dlen):
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    result = np.empty([Vdim, nbin.prod()], float)

    if statistic == 'mean':
        result.fill(np.nan)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic == 'median':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in range(Vdim):
                result[vv, i] = np.median(values[vv, binnumbers == i])

    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if(expand_binnumbers and Ndim > 1):
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`reulst`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    return result


def image_to_plot(a, upper=True):
    """
    Convert an array from image order convention
    with shape (nrows, ncols) starting upper left
    to array in ploting order convention
    with shape (nx, ny) and starting lower left
    """
    if upper:
        a = np.flip(a, axis=0)
    a = np.transpose(a)
    return a


def plot_to_image(a, upper=True):
    """
    Convert an array from ploting order convention
    with shape (nx, ny) and starting lower left
    to array in image order convention
    with shape (nrows, ncols) starting upper left
    """
    a = np.transpose(a)
    if upper:
        a = np.flip(a, axis=0)
    return a


def center_to_edge(centers):
    delta = centers[1] - centers[0]
    edges = np.insert(centers + delta/2, 0, centers[0] - delta/2)

    return edges


if __name__ == '__main__':
    print('wradlib: Calling module <util> as main...')
