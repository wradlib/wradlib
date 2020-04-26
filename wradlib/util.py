#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
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
           'find_bbox_indices', 'get_raster_origin', 'calculate_polynomial',
           'derivate']
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


def _pad_array(data, pad, mode='edge', **kwargs):
    """Returns array with padding added along last dimension.
    """
    pad_width = [(0,)] * (data.ndim - 1) + [(pad,)]
    if mode in ["maximum", "mean", "median", "minimum"]:
        kwargs['stat_length'] = kwargs.pop('stat_length', pad)
    data = np.pad(data, pad_width, mode=mode, **kwargs)
    return data


def _rolling_dim(data, window):
    """Return array with rolling dimension of window-length added.
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape,
                                           strides=strides)


def _polyfit_1d(rhs, order=1, method='lstsq'):
    """ Calculates polyfit over last dimension of rhs.

    Calculates lhs from size of last dimension of rhs. Only method='lstsq' is
    multidimensional. The nan-methods only work on a single system. Hence the
    apply_along_axis.
    """
    shape = rhs.shape
    lhs = np.vander(np.arange(shape[-1], dtype=rhs.dtype), order + 1)
    rhs = rhs.reshape((-1, shape[-1])).T

    if method == 'lstsq':
        out = np.linalg.lstsq(lhs, rhs, rcond=None)[0][0]
    elif method == 'lstsq_nan':
        out = np.apply_along_axis(_nanpolyfit_lstsq, 0, rhs, lhs)
    elif method == 'matrix_inv_nan':
        out = np.apply_along_axis(_nanpolyfit_matrix_inv, 0, rhs, lhs)
    else:
        raise ValueError(f"wradlib: unknown method value {method}")

    return out.reshape(shape[:-1])


def _nanpolyfit_lstsq(y, x):
    """Single polyfit by least squares considering NaN.
    """
    mask = np.isnan(y)
    out, _, _, _ = np.linalg.lstsq(x[~mask, :], y[~mask], rcond=None)
    return out[0]


def _nanpolyfit_matrix_inv(y, x):
    """Single polyfit by matrix inversion considering NaN.
    """
    mask = np.isnan(y)
    x = x[~mask]
    out = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y[~mask]))
    return out[0]


def _polyfit_covariance(y, skipna=False):
    shape = y.shape
    y = y.reshape((-1, y.shape[-1]))
    idx = np.argsort(y, axis=-1)
    y = np.sort(y, axis=-1)
    n = y.shape[-1]
    x = np.arange(n)
    x = x[idx]
    x = x.T
    y = y.T
    cov = np.sum((x - x.mean(axis=0)) * (y - y.mean(axis=0)), axis=0) / n
    out = cov / (x.std(axis=0)**2)
    return out.T.reshape(shape[:-1])


def _lanczos_differentiator(winlen):
    """Returns Lanczos Differentiator.
    """
    m = (winlen - 1) / 2
    denom = m * (m + 1.) * (2 * m + 1.)
    k = np.arange(1, m + 1)
    f = 3 * k / denom
    return np.r_[f[::-1], [0], -f]


def _derivate(data, window, method='lanczos_conv'):
    """Calculates derivative of prepared data.

    :param data:
    :param window:
    :param method:
    :return:
    """
    if method not in ['lanczos_conv', 'lanczos_dot', 'polyfit', 'covariance',
                      'slow']:
        raise ValueError(f"wradlib: unknown method value {method}")

    if 'lanczos' in method:
        win = _lanczos_differentiator(window)

    if method == 'lanczos_conv':
        # we use constant nan padding here, since more sophisticated padding
        # can be done outside the _derivate helper function
        out = ndimage.filters.convolve1d(data, win, axis=-1, mode='constant')
    else:
        data = _rolling_dim(data, window)
        if method == 'lanczos_dot':
            out = np.dot(data, win * -1)
        if method in ['slow', 'polyfit']:
            out = _polyfit_1d(data, method='lstsq')
        if method == 'covariance':
            out = _polyfit_covariance(data)

    return out


def derivate(data, winlen=7, method='lanczos_conv', skipna=False, **kwargs):
    """Calculates derivative of data.

    In normal operation the method ('lanczos_conv') uses convolution
    to estimate the derivative using Low-noise Lanczos differentiators.
    The equivalent method ('lanczos_dot') uses dot-vector sum product.

    For further reading please see `Differentiation by integration using \
    orthogonal polynomials, a survey <https://arxiv.org/pdf/1102.5219>`_ \
    and `Low-noise Lanczos differentiators \
    <http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/\
lanczos-low-noise-differentiators/>`_.

    The results are very similar to the moving window linear
    regression (`polyfit`) method, but the former are *much* faster.

    All methods will return NaNs in case at least one value in the moving
    window is NaN.

    If `skipna=True` the location of NaN results are treated by using local
    linear regression by matrix inversion where enough valid neighbouring data
    is available.

    Before applying the actual derivation calculation the data is padded along
    the derivation dimension.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        multi-dimensional array, note that the derivation dimension must be the
        last dimension of the input array.
    winlen : int
        Width of the derivation window .
    method : str
        Defaults to 'lanczos_conv'. Can take one of 'lanczos_dot', 'polyfit'.
    skipna : bool
        Defaults to False. It true, treat NaN results by applying method2.

    Keyword Arguments
    -----------------
    method2 : str
        Defaults to 'matrix_inv_nan'. Can also take 'lstsq_nan'.
    min_period : int
        Defaults to winlen // 2.
    pad_mode : str
        Defaults to 'edge'. See :func:`numpy:numpy.pad`.
    pad_kwargs : dict
        Keyword arguments for padding, see :func:`numpy:numpy.pad`

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        array of derivates with the same shape as data
    """
    assert (winlen % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."

    shape = data.shape
    data = data.reshape((-1, shape[-1]))

    # Make really sure winlen is an integer
    winlen = int(winlen)

    # append padding on derivation dimension
    pad = winlen // 2

    # pad data using pad_mode
    pad_kwargs = kwargs.pop('pad_kwargs', {})
    pad_kwargs['mode'] = kwargs.pop('pad_mode', 'edge')
    data_pad = _pad_array(data, pad, **pad_kwargs)

    # calculate derivation using winlen and method
    out = _derivate(data_pad, winlen, method=method)
    # strip padding for convolution method
    if method == 'lanczos_conv':
        out = out[..., pad:-pad]

    # NaN treatment
    if skipna:
        min_period = kwargs.pop('min_period', pad)
        assert min_period >= 1, "min_period need to be >= 1."
        method2 = kwargs.pop('method2', 'matrix_inv_nan')

        # find remaining NaN values with valid neighbours
        invalid = np.isnan(out)
        if np.any(invalid):
            data = _rolling_dim(data_pad, winlen)
            data = data.reshape((-1, data.shape[-1]))
            valid = np.count_nonzero(~np.isnan(data), axis=-1) > min_period
            recalc = (valid & invalid.reshape(-1))
            # and interpolate using polyfit -> method2
            if np.any(recalc):
                out.flat[recalc] = _polyfit_1d(data[recalc], method=method2)

    return out.reshape(shape)


if __name__ == '__main__':
    print('wradlib: Calling module <util> as main...')
