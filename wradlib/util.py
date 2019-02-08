#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not
attributable to the other modules

.. autosummary::
   :nosignatures:
   :toctree: generated/

   from_to
   maximum_intensity_projection
   filter_window_polar
   filter_window_cartesian
   find_bbox_indices
   get_raster_origin
   calculate_polynomial
"""
import datetime as dt
from datetime import tzinfo, timedelta
import os

import numpy as np
from scipy.ndimage import filters
from osgeo import gdal, ogr
from scipy.signal import medfilt


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
        mod = __import__(module)
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


def maximum_intensity_projection(data, r=None, az=None, angle=None,
                                 elev=None, autoext=True):
    """Computes the maximum intensity projection along an arbitrary cut \
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

    from wradlib.georef import bin_altitude as bin_altitude

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
    hc = bin_altitude(x, elev, 0, re=6370040.)
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
    fun = getattr(filters, "%s_filter" % fun)
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


class UTC(tzinfo):
    """UTC implementation for tzinfo.

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
    data_file = os.path.join(get_wradlib_data_path(), relfile)
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
    return medfilt(x, kernel_size)


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
    return np.insert(diffs, 0, diff_begin, axis=axis)


def gradient_from_smoothed(x, n=5):
    """Computes gradient of smoothed data along final axis of an array
    """
    return gradient_along_axis(medfilt_along_axis(x, n)).astype("f4")


if __name__ == '__main__':
    print('wradlib: Calling module <util> as main...')
