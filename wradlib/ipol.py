#!/usr/bin/env python
# Copyright (c) 2011-2026, wradlib developers.
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

   {}
"""

__all__ = [
    "IpolBase",
    "Nearest",
    "Idw",
    "Linear",
    "OrdinaryKriging",
    "ExternalDriftKriging",
    "RectGrid",
    "RectBin",
    "QuadriArea",
    "interpolate",
    "interpolate_polar",
    "get_mapping",
    "IpolMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import re
from dataclasses import dataclass
from functools import reduce, singledispatch

import numpy as np
import scipy
import xarray as xr
from packaging.version import Version
from scipy import interpolate as sinterp
from scipy import ndimage, spatial, special, stats

from wradlib import georef, util, zonalstats
from wradlib.util import XarrayMethods, docstring


class MissingSourcesError(Exception):
    """Is raised in case no source coordinates are available for interpolation."""


class MissingTargetsError(Exception):
    """Is raised in case no interpolation targets are available."""


class IpolBase:
    """
    IpolBase(src, trg)

    The base class for interpolation in N dimensions.
    Provides the basic interface for all other classes.

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    """

    def __init__(self, src, trg, **kwargs):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        self.numsources = len(src)
        self.numtargets = len(trg)

    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsources, ...)
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
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float

        """
        if len(vals) != self.numsources:
            raise ValueError(
                f"Length of value array {len(vals)} does not correspond to number "
                f"of source points {self.numsources}"
            )
        self.valsshape = vals.shape
        self.valsndim = vals.ndim

    def _make_coord_arrays(self, x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)

        Parameters
        ----------
        x : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and
            the length of the ndarray corresponding to the number of points

        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 2:
                pass
            else:
                raise TypeError("Cannot deal wih 3-d arrays, yet.")
        return x

    def _make_2d(self, vals):
        """Reshape increase number of dimensions of vals if smaller than 2,
        appending additional dimensions (as opposed to the atleast_nd methods
        of numpy).

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            values who are to be reshaped to the right shape

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            if vals.shape==() [a scalar] output.shape will be (1, 1)
            if vals.shape==(npt, ) output.shape will be (npt, 1)
            if vals.ndim > 1 vals will be returned as is
        """
        if vals.ndim < 2:
            # ndmin might be 0, so we get it to 1-d first
            # then we add an axis as we assume that
            return np.atleast_1d(vals)[:, np.newaxis]
        else:
            return vals


def _query_tree(tree, trg, **kwargs):
    if Version(scipy.__version__) < Version("1.6"):
        kwargs.setdefault("n_jobs", -1)
    else:
        kwargs.setdefault("workers", -1)
    dists, ix = tree.query(trg, **kwargs)
    # avoid bug, if there is only one neighbor at all
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        ix = ix[:, np.newaxis]
    return dists, ix


def _create_tree(src, **kwargs):
    # plant a tree, use unbalanced tree as default
    kwargs.update(balanced_tree=kwargs.pop("balanced_tree", False))
    return spatial.cKDTree(src, **kwargs)


def _call_nearest(vals, ix, dists, n=1, maxdist=None, **kwargs):
    # get first neighbour
    trgvals = vals[ix[:, 0]]
    dist_n = dists[..., 0].copy()

    # iteratively fill NaN with next neighbours
    isnan = np.isnan(trgvals)
    if n > 1 & np.count_nonzero(isnan):
        for i in range(n - 1):
            nanidx = np.argwhere(isnan)[..., 0]
            trgvals[isnan] = vals[ix[:, i + 1]][isnan]
            dist_n[nanidx] = dists[..., i + 1][nanidx]
            isnan = np.isnan(trgvals)
            if not np.count_nonzero(isnan):
                break

    if maxdist is None:
        return trgvals
    else:
        return np.where(dist_n > maxdist, np.nan, trgvals)


class Nearest(IpolBase):
    """
    Nearest(src, trg)

    Nearest-neighbour interpolation in N dimensions.

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims) or cKDTree object
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    remove_missing : int
        Number of neighbours to consider in the presence of NaN, defaults to 0.

    Keyword Arguments
    -----------------
    **kwargs : dict
        keyword arguments of ipclass (see class documentation)

    Examples
    --------
    See :doc:`notebooks:notebooks/interpolation/interpolation`.

    Note
    ----
    Uses :class:`scipy:scipy.spatial.cKDTree`
    """

    def __init__(self, src, trg, *, remove_missing=0, **kwargs):
        query_kwargs = kwargs.pop("query_kwargs", {})

        if isinstance(src, spatial.cKDTree):
            self.tree = src
        else:
            src = self._make_coord_arrays(src)
            if len(src) == 0:
                raise MissingSourcesError
            self.tree = _create_tree(src, **kwargs)

        self.numsources = self.tree.n

        trg = self._make_coord_arrays(trg)
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError

        self.nnearest = remove_missing + 1

        # query tree
        self.dists, self.ix = _query_tree(
            self.tree, trg, k=self.nnearest, **query_kwargs
        )

    def __call__(self, vals, *, maxdist=None):
        """
        Evaluate interpolator for values given at the source points.

        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points, ) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        maxdist : float
            the maximum distance up to which an interpolated values is
            assigned - if maxdist is exceeded, np.nan will be assigned
            If maxdist==None, values will be assigned everywhere

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints,...)

        """
        self._check_shape(vals)

        return _call_nearest(
            vals, self.ix, self.dists, n=self.nnearest, maxdist=maxdist
        )


def _interpolate_mapping(src, trg, **kwargs):
    """
    Interpolate a Dataset or DataArray using precomputed KDTree dataset.

    Parameters
    ----------
    src : :class:`xarray:xarray.Dataset' or :class:`xarray:xarray.DataArray'
        Dataset or DataArray to interpolate. Expected dims: (eg. azimuth, range)
    trg : :class:`xarray:xarray.Dataset'
        KDTree output with 'ix' indices, 'dists' distances and 'x','y' coordinates

    Returns
    -------
    :class:`xarray:xarray.Dataset' or :class:`xarray:xarray.DataArray'
        Interpolated data on target grid (y, x)
    """
    method = kwargs.get("method")
    METHODS = dict(
        nearest=_call_nearest,
        inverse_distance=_call_inverse_distance_weighting,
    )
    kwargs.update({"func": METHODS[method]})

    src_coords = _normalize_coords(kwargs.get("src_coords"))
    trg_coords = _normalize_coords(kwargs.get("trg_coords"))

    def _get_core_coords(obj, coords):
        x = coords.x
        y = coords.y

        x_coord = obj.coords[x]
        y_coord = obj.coords[y]

        if x_coord.dims == y_coord.dims:
            npoints = x_coord.dims
        else:
            npoints = (y_coord.dims[0], x_coord.dims[0])
        return x_coord, y_coord, npoints

    _, _, npoints1 = _get_core_coords(src, src_coords)
    trg_x_coord, trg_y_coord, npoints2 = _get_core_coords(trg, trg_coords)

    # separate variables that will be applied with ufunc
    if isinstance(src, xr.Dataset):
        src_to_interp, keep = util.get_apply_ufunc_variables(src, npoints1)
    else:
        src_to_interp = src

    # stack source and index arrays
    src_stacked = src_to_interp.stack(npoints1=npoints1)
    trg_stacked = trg.stack(npoints2=npoints2)

    x = trg.ix.sizes[npoints2[1]]
    y = trg.ix.sizes[npoints2[0]]
    sizes = dict(x=x, y=y)

    # vectorization needed?
    vectorize = not set(src_stacked.dims) <= {"npoints1"}

    # maxdist
    # hack: preventing interpolation outside domain
    maxdist = np.sqrt(
        (trg_x_coord[1] - trg_x_coord[0]) ** 2 + (trg_y_coord[1] - trg_y_coord[0]) ** 2
    ).values * np.sqrt(2)
    kwargs.setdefault("maxdist", maxdist)

    # determine output dtypes
    if isinstance(src_stacked, xr.DataArray):
        output_dtypes = [src_stacked.dtype]
    else:
        output_dtypes = [src_stacked[list(src_stacked.data_vars.keys())[0]].dtype]

    def wrapper(src_vals, ix, dists, **kwargs):
        """wrapper around interpolator numpy implementation
        src_vals : (npoints1)
        ix       : (npoints2, k)
        returns  : (y, x)
        """
        x = kwargs.pop("x")
        y = kwargs.pop("y")
        func = kwargs.pop("func")
        out = func(src_vals, ix, dists, **kwargs)
        out = out.reshape((y, x))
        return out

    kwargs.update(sizes)
    out = xr.apply_ufunc(
        wrapper,
        src_stacked,
        trg_stacked.ix,
        trg_stacked.dists,
        input_core_dims=[["npoints1"], ["npoints2", "k"], ["npoints2", "k"]],
        output_core_dims=[npoints2],
        vectorize=vectorize,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=output_dtypes,
        output_sizes=sizes,
        on_missing_core_dim="copy",
    )

    # restore attributes and merge with kept variables if Dataset
    if isinstance(src, xr.Dataset):
        out = xr.merge([out, keep])
    else:
        out.attrs = src.attrs
        out.name = src.name

    # re-add x, y coordinates from trg
    out = out.assign_coords(x=trg_x_coord, y=trg_y_coord)

    return out


@dataclass(frozen=True)
class Coords:
    x: str = "x"
    y: str = "y"


def _normalize_coords(coords):
    if coords is None:
        return Coords()
    if isinstance(coords, Coords):
        return coords
    if isinstance(coords, dict):
        return Coords(**coords)
    raise TypeError("coords must be Coords or dict")


def get_mapping(src, trg, src_coords=None, trg_coords=None, **kwargs):
    """
    Create :class:`xarray:xarray.Dataset` with KDTree indices and distances derived from src and trg.

    Parameters
    ----------
    src: :class:`xarray:xarray.Dataset` or :class:`xarray:xarray.DataArray`
        Source data containing spatial coordinates. It should have two dimensions,
        typically representing x (azimuth, range) and y (azimuth, range).
    trg: :class:`xarray:xarray.Dataset` or :class:`xarray:xarray.DataArray`
        Target data with spatial coordinates to interpolate.
        It should also have two dimensions, designated by x and y.

    Keyword Arguments
    -----------------
    src_coords : dict or Coords, optional
        Mapping from canonical spatial roles to coordinate names in the source
        object. Keys must include `"x"` and `"y"`. For example:
        `{"x": "lon", "y": "lat"}`.
        If None, defaults to `{"x": "x", "y": "y"}`.
    trg_coords : dict or Coords, optional
        Mapping from canonical spatial roles to coordinate names in the target
        object. Keys must include `"x"` and `"y"`.
        If None, defaults to `{"x": "x", "y": "y"}`.
    **kwargs: additional keyword arguments
        Additional parameters that may be passed to the KDTree implementation and query.

    Returns
    -------
    :class:`xarray:xarray.Dataset`:
        ix - indices
        dists - distances
    """

    tree_kwargs = util.get_keys(
        kwargs, ["leafsize", "compact_nodes", "copy_data", "balanced_tree", "boxsize"]
    )
    tree_kwargs.setdefault("balanced_tree", False)
    query_kwargs = util.get_keys(
        kwargs, ["k", "eps", "p", "distance_upper_bound", "workers"]
    )
    query_kwargs.setdefault("k", 1)
    query_kwargs.setdefault("workers", -1)

    src_coords = _normalize_coords(src_coords)
    trg_coords = _normalize_coords(trg_coords)

    def _prep_coords(x, y, stack_dim):
        if x.ndim == 2:
            x_stacked = x.stack(**{stack_dim: x.dims})
            y_stacked = y.stack(**{stack_dim: y.dims})
        else:
            x2d, y2d = xr.broadcast(x, y)
            x_stacked = x2d.stack(**{stack_dim: x2d.dims})
            y_stacked = y2d.stack(**{stack_dim: y2d.dims})

        points = np.column_stack([x_stacked.values, y_stacked.values])
        return points, x_stacked, y_stacked

    # coordinate prep
    src_points, src_x_stacked, src_y_stacked = _prep_coords(
        src.coords[src_coords.x], src.coords[src_coords.y], stack_dim="npoints"
    )
    trg_points, trg_x_stacked, trg_y_stacked = _prep_coords(
        trg.coords[trg_coords.x], trg.coords[trg_coords.y], stack_dim="npoints2"
    )

    # build tree
    tree = spatial.cKDTree(src_points, **tree_kwargs)
    dists, ix = _query_tree(tree, trg_points, **query_kwargs)

    # return ix/dists as Dataset
    out = xr.Dataset(
        {
            "ix": (["npoints2", "k"], ix),
            "dists": (["npoints2", "k"], dists),
        },
        coords={
            "npoints2": trg_x_stacked["npoints2"],
            "k": np.arange(query_kwargs["k"]),
        },
    )

    out = out.unstack("npoints2")
    out.attrs = dict(
        source="wradlib",
        model="kdtree",
        tree_kwargs=tree_kwargs,
        query_kwargs=query_kwargs,
    )

    return out


def _call_inverse_distance_weighting(
    vals, ix, dists, idw_p=2.0, remove_missing=False, maxdist=None, **kwargs
):
    weights = 1.0 / dists**idw_p

    # if maxdist isn't given, take the maximum distance
    if maxdist is not None:
        outside = dists > maxdist
        weights[outside] = 0

    # take care of point coincidence
    weights[np.isposinf(weights)] = 1e12

    # shape handling (time, ensemble etc)
    wshape = weights.shape
    weights.shape = wshape + ((vals.ndim - 1) * (1,))

    # expand vals to trg grid
    trgvals = vals[ix]

    # nan handling
    if remove_missing:
        isnan = np.isnan(trgvals)
        weights = np.broadcast_to(weights, isnan.shape)
        masked_weights = np.ma.array(weights, mask=isnan)

        interpol = np.nansum(weights * trgvals, axis=1) / np.sum(masked_weights, axis=1)
    else:
        interpol = np.sum(weights * trgvals, axis=1) / np.sum(weights, axis=1)

    return interpol


class Idw(IpolBase):
    """
    Idw(src, trg, nnearest=4, p=2.)

    Inverse distance weighting interpolation in N dimensions.

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims) of cKDTree object
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    nnearest : int
        max. number of neighbours to be considered
    p : float
        inverse distance power used in 1/dist**p
    remove_missing : bool
        If True masks NaN values in the data values, defaults to False


    Keyword Arguments
    -----------------
    **kwargs : dict
        keyword arguments of ipclass (see class documentation)

    Examples
    --------
    See :doc:`notebooks:notebooks/interpolation/interpolation`.

    Note
    ----
    Uses :class:`scipy:scipy.spatial.cKDTree`

    """

    def __init__(self, src, trg, *, nnearest=4, p=2.0, remove_missing=False, **kwargs):
        query_kwargs = kwargs.pop("query_kwargs", {})
        if isinstance(src, spatial.cKDTree):
            self.tree = src
        else:
            src = self._make_coord_arrays(src)
            if len(src) == 0:
                raise MissingSourcesError
            self.tree = _create_tree(src, **kwargs)

        self.numsources = self.tree.n

        trg = self._make_coord_arrays(trg)
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError

        if nnearest > self.numsources:
            util.warn(
                "wradlib.ipol.Idw: `nnearest` is larger than number of "
                f"source points and is set to {self.numsources} corresponding to the "
                "number of source points.",
                UserWarning,
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest

        self.remove_missing = remove_missing

        self.p = p
        # query tree
        self.dists, self.ix = _query_tree(
            self.tree, trg, k=self.nnearest, **query_kwargs
        )

    def __call__(self, vals, *, maxdist=None):
        """
        Evaluate interpolator for values given at the source points.

        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points, ) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate

        maxdist : float
            the maximum distance up to which points will be included into the
            interpolation calculation

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints,...)

        """
        self._check_shape(vals)

        return _call_inverse_distance_weighting(
            vals,
            self.ix,
            self.dists,
            idw_p=self.p,
            remove_missing=self.remove_missing,
            maxdist=maxdist,
        )


class Linear(IpolBase):
    """
    Interface to the :class:`scipy:scipy.interpolate.LinearNDInterpolator`
    class.

    We provide this class in order to achieve a uniform interface for all
    Interpolator classes

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.

    Examples
    --------
    See :doc:`notebooks:notebooks/interpolation/interpolation`.
    """

    def __init__(self, src, trg, *, remove_missing=False):
        self.src = self._make_coord_arrays(src)
        self.trg = self._make_coord_arrays(trg)
        self.remove_missing = remove_missing
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        self.numsources = len(src)
        if self.numsources == 0:
            raise MissingSourcesError

    def __call__(self, vals, *, fill_value=np.nan):
        """
        Evaluate interpolator for values given at the source points.

        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points, ) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        fill_value : float
            is needed if linear interpolation fails; defaults to np.nan

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints,...)

        """
        self._check_shape(vals)
        isnan = np.isnan(vals)
        if self.remove_missing & np.count_nonzero(isnan):
            ip = sinterp.LinearNDInterpolator(
                self.src[~isnan, ...], vals[~isnan], fill_value=fill_value
            )
        else:
            ip = sinterp.LinearNDInterpolator(self.src, vals, fill_value=fill_value)
        return ip(self.trg)


class RectGridBase:
    """Rectangular Grid - base class

    Parameters
    ----------
    grid : :class:`numpy:numpy.ndarray` of floats
        3d array of shape (..., 2)
        The points defining the regular grid in n dimensions.
    points : :class:`numpy:numpy.ndarray` of floats
        Array of shape (..., 2)
        The sample point coordinates.
    """

    def __init__(self, grid, points):
        self._upper = None
        self._xdim = None
        self._ydim = None
        self._image = None
        self._upper = None
        self._is_grid = None
        self._ipol_grid = None
        self._ipol_points = None
        self._grid = np.array(grid)
        self._points = np.array(points)

    @property
    def is_grid(self):
        if self._is_grid is None:
            self._is_grid = self.grid.ndim == 3 and self.grid.shape[2] == 2
            if not self._is_grid:
                raise ValueError(
                    f"Grid Shape mismatch, expected (N, M, 2), but got {self.grid.shape}."
                )
        return self._is_grid

    @property
    def ydim(self):
        if self._ydim is None:
            self._ydim = 0 if self.image else 1
        return self._ydim

    @property
    def xdim(self):
        if self._xdim is None:
            self._xdim = 1 if self.image else 0
        return self._xdim

    @property
    def image(self):
        if self._image is None:
            self._image = self.grid[0, 0, 1] == self.grid[0, 1, 1]
        return self._image

    @property
    def upper(self):
        if self._upper is None:
            self._upper = (
                np.diff(np.take(self.grid[..., 1], 0, axis=self.xdim)[0:2])[0] < 0
            )
        return self._upper

    @property
    def grid(self):
        return self._grid

    @property
    def points(self):
        return self._points

    @property
    def ipol_grid(self):
        if self._ipol_grid is None:
            self._ipol_grid = self._get_grid_dims()
        return self._ipol_grid

    @property
    def ipol_points(self):
        if self._ipol_points is None:
            self._ipol_points = self._get_points()
        return self._ipol_points

    def _get_grid_dims(self):
        grd = self.grid
        if self.image:
            grd = np.flip(grd, -1)
        if self.upper:
            grd = np.flip(grd, self.ydim)
        grd_dim0 = np.take(grd[..., 0], 0, axis=1)
        grd_dim1 = np.take(grd[..., 1], 0, axis=0)
        return grd_dim0, grd_dim1

    def _get_points(self):
        pts = self.points
        if self.image:
            pts = np.flip(pts, -1)
        return pts.reshape((-1, 2))


class RectGrid(RectGridBase):
    """Interpolation on a 2d grid in arbitrary dimensions.

    The source data must be defined on a regular grid, the grid spacing
    however may be uneven. Linear, nearest-neighbour and spline
    interpolation are supported.

    Based on :py:func:`scipy:scipy.interpolate.interpn`, uses:
    - `nearest` :py:class:`scipy:scipy.interpolate.RegularGridInterpolator`
    - `linear` :py:class:`scipy:scipy.interpolate.RegularGridInterpolator`
    - `splinef2d` :py:class:`scipy.interpolate.RectBivariateSpline`

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        3d array of shape (..., 2)
        The points defining the regular grid in n dimensions.
    trg : :class:`numpy:numpy.ndarray`
        Array of shape (..., ndim)
        The coordinates to sample the gridded data at

    Keyword Arguments
    -----------------
    method : str
        Method of interpolation used, defaults to 'linear'.

    """

    def __init__(self, src, trg, *, method="linear"):
        super().__init__(src, trg)
        self.method = method

    def __call__(self, values, **kwargs):
        """Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        values : :class:`numpy:numpy.ndarray`
            Values at the source points which to interpolate, shape (num src pts, ...)

        Returns
        -------
        result : :class:`numpy:numpy.ndarray`
            Target values with shape (num trg pts, ...)

        """

        # override bounds_error
        kwargs["bounds_error"] = kwargs.pop("bounds_error", False)
        kwargs["method"] = kwargs.pop("method", self.method)

        # need to flip ydim if grid origin is 'upper'
        if self.upper:
            values = np.flip(values, self.ydim)

        result = sinterp.interpn(self.ipol_grid, values, self.ipol_points, **kwargs)

        return result.reshape(self.points.shape[:-1])


class RectBin(RectGridBase):
    """Bin points values to regular grid cells
       e.g. for transforming a radar sweep to a raster image

    Based on :py:func:`scipy:scipy.stats.binned_statistic_dd`

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        data point coordinates of the source points with shape (..., ndims).
    trg : :class:`numpy:numpy.ndarray`
        rectangular grid coordinates (center) with shape (..., 2)
    fill : array of boolean with same shape as src
        used to fill nan with nearest neighbour value
        e.g. the radar mask
    """

    def __init__(self, src, trg, fill=None):
        super().__init__(trg, src)
        self._binned_stats = False
        src = src.reshape(-1, 2)
        trg = trg.reshape(-1, 2)
        self.fill = fill
        if self.fill is not None:
            self.nearest = Nearest(src, trg)

    @property
    def binned_stats(self):
        return self._binned_stats

    @binned_stats.setter
    def binned_stats(self, value):
        self._binned_stats = value

    def _get_grid_dims(self):
        dims = super()._get_grid_dims()
        return [util.center_to_edge(x) for x in dims]

    def __call__(self, values, **kwargs):
        """Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        values : :class:`numpy:numpy.ndarray`
            Values at the source points which to interpolate, shape (num src pts, ...)
        kwargs : dict
            keyword arguments passed to scipy.stats.binned_statistic_dd

        Returns
        -------
        stat : :class:`numpy:numpy.ndarray`
            Target values with shape (num trg pts, ...)
        """

        kwargs.setdefault("statistic", np.nanmean)

        # reshape into flat array
        values = values.reshape(-1)

        if not self.binned_stats or Version(scipy.__version__) < Version("1.4"):
            result = stats.binned_statistic_dd(
                self.ipol_points,
                values,
                bins=self.ipol_grid,
                **kwargs,
            )
            self.binned_stats = result
        else:
            result = stats.binned_statistic_dd(
                self.ipol_points,
                values,
                binned_statistic_result=self.binned_stats,
                **kwargs,
            )
        stat = result.statistic

        # need to flip ydim if grid origin is 'upper'
        if self.upper:
            stat = np.flip(stat, self.ydim)

        # fill gaps with nearest values
        if self.fill is not None:
            values = values.reshape(-1)
            nearest = self.nearest(values)
            nearest = nearest.reshape(stat.shape)
            nearest[~self.fill] = np.nan
            gaps = np.isnan(stat)
            stat[gaps] = nearest[gaps]

        return stat


class PolyArea:
    """Map values representing polygons to another polygons

    Based on :mod:`wradlib.zonalstats`

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Source grid edge coordinates with shape (..., 5, 2).
    trg : :class:`numpy:numpy.ndarray`
        Target grid edge coordinates with shape (..., 5, 2).
    """

    def __init__(self, src, trg, **kwargs):
        self.shape = trg.shape[:-2]

        src = src.reshape((-1, 5, 2))
        trg = trg.reshape((-1, 5, 2))

        zd = zonalstats.ZonalDataPoly(src, trg=trg, **kwargs)
        self.obj = zonalstats.ZonalStatsPoly(zd)

    def __call__(self, values):
        """Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        values : :class:`numpy:numpy.ndarray`
            Values representing the source cells, shape corresponding to src

        Returns
        -------
        result : :class:`numpy:numpy.ndarray`
            Values representing the target cells, shape corresponding to trg
        """

        values = values.ravel()
        result = self.obj.mean(values)

        return result.reshape(self.shape)


class QuadriArea(PolyArea):
    """Map values representing quadrilateral grid cells to another quadrilateral grid.

    Based on :mod:`wradlib.zonalstats`

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        Source grid edge coordinates with shape (n+1, m+1, 2).
    trg : :class:`numpy:numpy.ndarray`
        Target grid edge coordinates with shape (o+1, p+1, 2).
    """

    def __init__(self, src, trg, **kwargs):
        src = georef.rect.grid_to_polyvert(src)
        trg = georef.rect.grid_to_polyvert(trg)
        super().__init__(src, trg, **kwargs)


class IpolChain(IpolBase):
    """Apply successive interpolation methods.

    Parameters
    ----------
    interpolators: list
        list of interpolators (IpolBase) to apply successivly
    """

    def __init__(self, interpolators):
        self.interpolators = interpolators

    def __call__(self, values, **kwargs):
        """Evaluate interpolator for values given at the source points.

        Parameters
        ----------
        values : :class:`numpy:numpy.ndarray`
            Values at src points which to interpolate with shape (num src pts, ...)

        Returns
        -------
        result : class:`numpy:numpy.ndarray`
            Values at the trg points with shape (num trg pts, ...)
        """

        first = self.interpolators.pop(0)
        result = first(values, **kwargs)

        for interpolator in self.interpolators:
            temp = interpolator(values, **kwargs)
            bad = np.isnan(result)
            result[bad] = temp[bad]

        return result


# -----------------------------------------------------------------------------
# Covariance routines needed for Kriging
# -----------------------------------------------------------------------------
def parse_covariogram(cov_model):
    """ """
    patterns = [
        re.compile(r"([\d\.]+) Nug\(([\d\.]+)\)"),  # nugget
        re.compile(r"([\d\.]+) Lin\(([\d\.]+)\)"),  # linear
        re.compile(r"([\d\.]+) Sph\(([\d\.]+)\)"),  # spherical
        re.compile(r"([\d\.]+) Exp\(([\d\.]+)\)"),  # exponential
        re.compile(r"([\d\.]+) Gau\(([\d\.]+)\)"),  # gaussian
        re.compile(r"([\d\.]+) Mat\(([\d\.]+)\)\^([\d\.]+)"),  # matern
        re.compile(r"([\d\.]+) Pow\(([\d\.]+)\)"),  # power
        # cauchy
        re.compile(r"([\d\.]+) " r"Cau\(([\d\.]+)\)\^([\d\.]+)\^([\d\.]+)"),
    ]

    cov_funs = [
        cov_nug,
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
    subparts = cov_model.split("+")
    # then analyse subparts
    for _i, subpart in enumerate(subparts):
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
    return np.where(h <= rng, sill, 0.0)


def cov_exp(h, sill=1.0, rng=1.0):
    """exponential type covariance function"""
    h = np.asanyarray(h)
    return sill * (np.exp(-h / rng))


def cov_sph(h, sill=1.0, rng=1.0):
    """spherical type covariance function"""
    h = np.asanyarray(h)
    return np.where(h < rng, sill * (1.0 - 1.5 * h / rng + h**3 / (2 * rng**3)), 0.0)


def cov_gau(h, sill=1.0, rng=1.0):
    """gaussian type covariance function"""
    h = np.asanyarray(h)
    return sill * np.exp(-(h**2) / rng**2)


def cov_lin(h, sill=1.0, rng=1.0):
    """linear covariance function"""
    h = np.asanyarray(h)
    return np.where(h < rng, sill * (-h / rng + 1.0), 0.0)


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
        kv = special.kv
        # Gamma function
        tau = special.gamma

        fac1 = h / rng * 2.0 * np.sqrt(shp)
        fac2 = tau(shp) * 2.0 ** (shp - 1.0)

        c = np.where(h != 0, sill * 1.0 / fac2 * fac1**shp * kv(shp, fac1), sill)

    return c


def cov_pow(h, sill=1.0, rng=1.0):
    """power law covariance function"""
    h = np.asanyarray(h)
    return sill - h**rng


def cov_cau(h, sill=1.0, rng=1.0, alpha=1.0, beta=1.0):
    """
    cauchy covariance function.

    alpha >0 & <=2 ... shape parameter
    beta >0 ... parameterizes long term memory
    """
    h = np.asanyarray(h).astype("float")
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
    indicates a separation distance after which the covariance drops
    close to zero) a sometimes additional parameters governing the shape
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
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    cov : str
        covariance (variogram) model string in the syntax ``gstat``
        uses.
    nnearest : int
        max. number of neighbours to be considered
    remove_missing : bool
        If True masks NaN values in the data values, defaults to False


    Note
    ----
    The class calculates the Kriging weights during initialization, because
    these only depend on the configuration of the points.

    The call method is then only used to calculate estimated values at the
    target points based on those at the source points. Therefore, the main
    computational load is experienced during initialization. This behavior is
    different from that of the Idw or Nearest Interpolators.

    After initialization the estimation variance at each interpolation target
    may be retrieved from the attribute `estimation_variance`.

    Examples
    --------
    See :doc:`notebooks:notebooks/interpolation/interpolation`.
    """

    def __init__(
        self,
        src,
        trg,
        cov="1.0 Exp(10000.)",
        *,
        nnearest=12,
        remove_missing=False,
        **kwargs,
    ):
        """ """
        if isinstance(src, spatial.cKDTree):
            self.tree = src
            self.src = self.tree.data
        else:
            self.src = self._make_coord_arrays(src)
            if len(src) == 0:
                raise MissingSourcesError
            # plant a tree, use unbalanced tree as default
            kwargs.update(balanced_tree=kwargs.pop("balanced_tree", False))
            self.tree = spatial.cKDTree(self.src, **kwargs)

        self.numsources = self.tree.n

        self.remove_missing = remove_missing

        self.trg = self._make_coord_arrays(trg)
        # remember some things
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError
        if nnearest > self.numsources:
            util.warn(
                "wradlib.ipol.OrdinaryKriging: `nnearest` is "
                "larger than number of source points and is "
                f"set to {self.numsources} corresponding to the "
                "number of source points.",
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest

        # query tree
        # scipy kwarg changed from version 1.6
        if Version(scipy.__version__) < Version("1.6"):
            query_kwargs = dict(n_jobs=-1)
        else:
            query_kwargs = dict(workers=-1)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest, **query_kwargs)
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
        """Sets up the kriging system for a configuration of source points."""
        var_matrix = self.cov_func(spatial.distance_matrix(src, src))

        ok_matrix = np.ones((len(src) + 1, len(src) + 1))

        ok_matrix[:-1, :-1] = var_matrix
        ok_matrix[-1, -1] = 0.0

        return ok_matrix

    def _krig_rhs(self, dists):
        """Sets up a right hand side of the kriging system given the distances
        of the target to the source points. To be used in conjunction with
        `_krig_matrix`."""
        rhs = self.cov_func(dists)
        ok_rhs = np.concatenate([rhs, [1.0]])

        return ok_rhs

    def _krige(self):
        """Sets up the kriging system and solves it in order to obtain the
        interpolation weights of ordinary kriging.
        Also calculates the kriging estimation variance from the results"""
        for dist, ix in zip(self.dists, self.ix, strict=True):
            matrix = self._krig_matrix(self.src[ix, :])
            rhs = self._krig_rhs(dist)
            weights = np.linalg.solve(matrix, rhs)
            self.weights.append(weights)
            self.estimation_variance.append(self.cov_func(0.0) - np.sum(weights * rhs))

    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.

        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points, ) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, numfields)
            Values at the source points from which to interpolate
            Several fields may be calculated at once by passing them
            along the second dimension.
            Only this second dimension is implemented. You'll have to
            reshape a more complex array for the function to work.

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints, numfields)

        """
        v = self._make_2d(vals)
        self._check_shape(v)

        # expand vals to trg grid
        trgvals = v[self.ix]

        # calculate estimator
        weights = np.array(self.weights)

        # nan handling
        if self.remove_missing:
            isnan = np.isnan(trgvals)
            weights = np.broadcast_to(weights[:, :-1][..., np.newaxis], isnan.shape)
            masked_weights = np.ma.array(weights, mask=isnan)

            interpol = np.nansum(masked_weights * trgvals, axis=1)
        else:
            interpol = np.sum(weights[:, :-1][..., np.newaxis] * trgvals, axis=1)

        if vals.ndim == 1:
            return interpol.ravel()
        else:
            return interpol


class ExternalDriftKriging(IpolBase):
    """
    ExternalDriftKriging(src, trg, cov='1.0 Exp(10000.)', nnearest=12,
                         drift_src=None, drift_trg=None)

    Kriging with external drift

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (nsrcpoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (ntrgpoints, ndims)
        Data point coordinates of the target points.
    cov : str
        covariance (variogram) model string in the syntax ``gstat``
        uses.
    nnearest : int
        max. number of neighbours to be considered
    src_drift : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (nsrcpoints, )
        values of the external drift at each source point
    trg_drift : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (ntrgpoints, )
        values of the external drift at each target point

    See Also
    --------
    :class:`~wradlib.ipol.OrdinaryKriging`

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

    Examples
    --------
    See :doc:`notebooks:notebooks/interpolation/interpolation`.
    """

    def __init__(
        self,
        src,
        trg,
        cov="1.0 Exp(10000.)",
        *,
        nnearest=12,
        src_drift=None,
        trg_drift=None,
        remove_missing=False,
        **kwargs,
    ):
        """ """
        if isinstance(src, spatial.cKDTree):
            self.tree = src
            self.src = self.tree.data
        else:
            self.src = self._make_coord_arrays(src)
            if len(src) == 0:
                raise MissingSourcesError
            # plant a tree, use unbalanced tree as default
            kwargs.update(balanced_tree=kwargs.pop("balanced_tree", False))
            self.tree = spatial.cKDTree(self.src, **kwargs)

        self.numsources = self.tree.n
        self.remove_missing = remove_missing
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
            util.warn(
                "wradlib.ipol.ExternalDriftKriging: `nnearest` is larger "
                f"than number of source points and is set to {self.numsources} "
                "corresponding to the number of source points.",
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest
        # query tree
        # scipy kwarg changed from version 1.6
        if Version(scipy.__version__) < Version("1.6"):
            query_kwargs = dict(n_jobs=-1)
        else:
            query_kwargs = dict(workers=-1)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest, **query_kwargs)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:, np.newaxis]
            self.ix = self.ix[:, np.newaxis]
        # parse covariogram function string
        self.cov_func = parse_covariogram(cov)
        self.weights = []
        self.estimation_variance = []

    def _krig_matrix(self, src, drift):
        """Sets up the kriging system for a configuration of source points."""
        # the basic covariance matrix
        var_matrix = self.cov_func(spatial.distance_matrix(src, src))
        # the extended matrix, initialized to ones
        edk_matrix = np.ones((len(src) + 2, len(src) + 2))

        # adding entries for the first lagrange multiplier for the ordinary
        # kriging part
        edk_matrix[:-2, :-2] = var_matrix
        edk_matrix[-2, -2] = 0.0

        # adding entries for the second lagrange multiplier for the  edk part
        edk_matrix[:-2, -1] = drift
        edk_matrix[-1, :-2] = drift
        edk_matrix[-2:, -1] = 0.0
        edk_matrix[-1, -2:] = 0.0

        return edk_matrix

    def _krig_rhs(self, dists, drift):
        """Sets up a right hand side of the kriging system given the distances
        of the target to the source points. To be used in conjunction with
        `_krig_matrix`."""
        rhs = self.cov_func(dists)
        edk_rhs = np.concatenate([rhs, np.array([1.0, drift])])

        return edk_rhs

    def _krige(self, src_drift, trg_drift):
        """Sets up the kriging system and solves it in order to obtain the
        interpolation weights of ordinary kriging.
        Also calculates the kriging estimation variance from the results"""
        all_weights = []
        estimation_variances = []
        for dist, ix, td in zip(self.dists, self.ix, trg_drift, strict=True):
            matrix = self._krig_matrix(self.src[ix, :], src_drift[ix])
            rhs = self._krig_rhs(dist, td)
            try:
                weights = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                weights = np.repeat(np.nan, len(rhs))
            all_weights.append(weights)
            estimation_variances.append(self.cov_func(0.0) - np.sum(weights * rhs))

        return all_weights, estimation_variances

    def __call__(self, vals, *, src_drift=None, trg_drift=None):
        """
        Evaluate interpolator for values given at the source points.

        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points, ) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, numfields)
            Values at the source points from which to interpolate
            Several fields may be calculated at once by passing them
            along the second dimension.
            Only this second dimension is implemented. You'll have to
            reshape a more complex array for the function to work.

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints, numfields)

        """
        if vals.ndim > 2:
            raise ValueError(
                f"`vals` nedd to be a 2D-array, but is of shape {vals.shape}."
            )
        v = self._make_2d(vals)
        self._check_shape(v)

        if src_drift is None:
            # check if we have data from __init__
            if self.src_drift is None:
                raise ValueError(
                    "`src_drift`-kwarg must be specified either on "
                    "initialization or when calling the interpolator."
                )
            src_drift = self.src_drift
        if trg_drift is None:
            # check if we have data from __init__
            if self.trg_drift is None:
                raise ValueError(
                    "`trg_drift`-kwarg must be specified either on "
                    "initialization or when calling the interpolator."
                )
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

            trgvals = v[self.ix]
            if self.remove_missing:
                isnan = np.isnan(trgvals)
                weights = np.broadcast_to(weights[:, :-2][..., np.newaxis], isnan.shape)
                masked_weights = np.ma.array(weights, mask=isnan)
                ip = np.nansum(masked_weights * trgvals, axis=1)
            else:
                ip = np.nansum(weights[:, :-2][..., np.newaxis] * trgvals, axis=1)
        # otherwise we need to set up and solve the kriging system for each
        # field individually
        else:
            ip = np.empty((self.trg.shape[0], v.shape[1]))
            if (v.shape[1] != src_d.shape[1]) or (v.shape[1] != trg_d.shape[1]):
                raise ValueError(
                    f"`vals` shape[1] ({v.shape[1]}) does not match `src` "
                    f"({src_d.shape[1]}) and `trg` ({trg_d.shape[1]})."
                )
            for i in range(v.shape[1]):
                wght, variances = self._krige(
                    src_d[:, i].squeeze(), trg_d[:, i].squeeze()
                )

                weights = np.array(wght)

                trgvals = v[self.ix, i]
                if self.remove_missing:
                    isnan = np.isnan(trgvals)
                    weights = np.broadcast_to(weights[:, :-2], isnan.shape)
                    masked_weights = np.ma.array(weights, mask=isnan)
                    ip[:, i] = np.nansum(masked_weights * trgvals, axis=1)
                else:
                    ip[:, i] = np.nansum(weights[:, :-2] * trgvals, axis=1)

                self.weights.append(weights)
                self.estimation_variance.append(variances)

        if vals.ndim == 1:
            return ip.ravel()
        else:
            return ip


# -----------------------------------------------------------------------------
# Wrapper functions
# -----------------------------------------------------------------------------
@singledispatch
def interpolate(src, trg, vals, ipclass, *args, **kwargs):
    """
    Convenience function to use the interpolation classes in an efficient way

    The interpolation classes in :mod:`wradlib.ipol` are computationally very
    efficient if they are applied on large multidimensional arrays of which
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
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    vals : :class:`numpy:numpy.ndarray`
        ndarray of float, shape (numsourcepoints, ...)
        Values at the source points which to interpolate
    ipclass : :class:`wradlib.ipol.IpolBase`
        A class which inherits from IpolBase.

    Other Parameters
    ----------------
    *args : list
        arguments of ipclass (see class documentation)

    Keyword Arguments
    -----------------
    **kwargs : dict
        keyword arguments of ipclass (see class documentation)

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
        ip = ipclass(src[ix_valid], trg, *args, **kwargs)
        result = ip(vals[ix_valid])
    elif vals.ndim == 2:
        # this implementation for 2 dimensions needs generalization
        ip = ipclass(src, trg, *args, **kwargs)
        result = ip(vals)
        nan_in_result = np.where(np.isnan(result))
        # nan_in_vals = np.where(np.isnan(vals))
        for i in np.unique(nan_in_result[-1]):
            ix_good = np.where(np.isfinite(vals[..., i]))[0]
            tmp = np.where(nan_in_result[-1] == i)[0]
            ix_broken_targets = nan_in_result[0][tmp]
            ip = ipclass(
                src[ix_good],
                trg[nan_in_result[0][np.where(nan_in_result[-1] == i)[0]]],
                *args,
                **kwargs,
            )
            tmp = ip(vals[ix_good, i].reshape((len(ix_good), -1)))
            result[ix_broken_targets, i] = tmp.ravel()
    else:
        if np.any(np.isnan(vals.ravel())):
            raise NotImplementedError(
                "At the moment, `interpolate` can only deal with NaN values in `vals` "
                "if `vals` has less than 3 dimension."
            )
        else:
            # if no NaN value are in <vals> we can safely apply the
            # ipclass as is
            ip = ipclass(src, trg, *args, **kwargs)
            result = ip(vals)
    return result


@interpolate.register(xr.DataArray)
@interpolate.register(xr.Dataset)
def _interpolate_xarray(src, trg, **kwargs):
    """
    Interpolate data from source coordinates to target coordinates using a chosen backend.

    This function provides a unified interface to multiple interpolation backends
    (:class:`scipy:scipy.spatial.KDTree`-based, :func:`scipy:scipy.interpolate.griddata` based, and :func:`scipy:scipy.ndimage.map_coordinates`) through a single
    `method` keyword. The behavior depends on the chosen backend and the dimensionality
    of the source and target data.

    Backends and method strings
    ---------------------------

    :class:`scipy:scipy.spatial.KDTree`-based methods (structured or unstructured points):
        - 'nearest'                : nearest neighbor
        - 'inverse_distance'       : inverse distance weighting (IDW)
        - 'ordinary_kriging'       : ordinary kriging

    :func:`scipy:scipy.interpolate.griddata` (arbitrary N-dimensional points):
        - 'griddata'               : uses SciPy default ('linear')
        - 'griddata_linear'        : linear interpolation
        - 'griddata_cubic'         : cubic interpolation

    :func:`scipy:scipy.ndimage.map_coordinates` (N-dimensional arrays with fractional coordinates):
        - 'map_coordinates'        : default order=3 (cubic)
        - 'map_coordinates_nearest': order=0
        - 'map_coordinates_linear' : order=1
        - 'map_coordinates_quadratic': order=2
        - 'map_coordinates_cubic'  : order=3
        - 'map_coordinates_quartic': order=4
        - 'map_coordinates_quintic': order=5

    Parameters
    ----------
    src : :class:`xarray:xarray.DataArray` or :class:`xarray:xarray.Dataset`
        Source data to be interpolated. Can be N-dimensional.
    trg : :class:`xarray:xarray.DataArray` or :class:`xarray:xarray.Dataset`
        Target coordinates. Expected shape and dimensionality depend on the backend:
            - :class:`scipy:scipy.spatial.KDTree`: unstructured or structured points (any number of dimensions)
            - :func:`scipy:scipy.interpolate.griddata`: N-dimensional points compatible with src
            - :func:`scipy:scipy.ndimage.map_coordinates`: fractional coordinates along each axis
    method : str
        Interpolation method / backend selection. See "Backends and method strings" above.
        Suffixes indicate backend-specific interpolation type (e.g., 'griddata_cubic').
    kwargs : dict
        Additional keyword arguments passed to the low-level interpolation routines.

    Returns
    -------
    :class:`xarray:xarray.DataArray` or :class:`xarray:xarray.Dataset`
        Interpolated data on the target coordinates. The output type matches the input.

    Notes
    -----
    - :class:`scipy:scipy.spatial.KDTree`-based methods automatically compute a mapping from source to target unless a
      precomputed mapping is provided.
    - :func:`scipy:scipy.interpolate.griddata` uses 'linear' by default if no specific method is provided.
    - :func:`scipy:scipy.ndimage.map_coordinates` uses order=3 (cubic) by default. User-supplied `order` overrides
      the method suffix; a warning is issued if both are provided.
    - The `method` keyword encodes both backend and interpolation type; the backend is inferred
      from the prefix ('nearest', 'griddata', 'map_coordinates', etc.).
    - Backend-specific kwargs take precedence over defaults when provided.
    - This function supports arbitrary N-dimensional data; polar-to-Cartesian is just one common use case.

    Examples
    --------
    >>> # KDTree nearest neighbor interpolation
    >>> out = src.wrl.ipol.interpolate(target, method='nearest')

    >>> # KDTree inverse distance weighting
    >>> out = src.wrl.ipol.interpolate(target, method='inverse_distance')

    >>> # SciPy griddata with cubic interpolation
    >>> out = src.wrl.ipol.interpolate(target, method='griddata_cubic', backend_kwargs={'fill_value': np.nan})

    >>> # SciPy map_coordinates with quintic spline interpolation
    >>> out = src.wrl.ipol.interpolate(target, method='map_coordinates_quintic')"""

    BACKENDS = [
        "nearest",
        "inverse_distance",
        "ordinary_kriging",
        "griddata",
        "map_coordinates",
    ]
    _kwargs = kwargs.copy()
    method = _kwargs.get("method")

    _kwargs.setdefault("src_coords", Coords())
    _kwargs.setdefault("trg_coords", Coords())

    backend = None
    backend_method = None
    for b in BACKENDS:
        if method.startswith(b):
            backend = b
            suffix = method[len(b) :]
            if suffix.startswith("_"):
                backend_method = suffix[1:]
            break

    if "griddata" == backend:
        if backend_method:
            _kwargs["method"] = backend_method
        return _griddata_xarray(src, trg, **_kwargs)
    elif "map_coordinates" == backend:
        if backend_method:
            if "order" in _kwargs:
                util.warn(
                    f"User-supplied 'order' ({_kwargs['order']}) overrides method='{backend_method}'"
                )
            else:
                ORDER_MAP = {
                    "nearest": 0,
                    "linear": 1,
                    "quadratic": 2,
                    "cubic": 3,
                    "quartic": 4,
                    "quintic": 5,
                }
            _kwargs["order"] = ORDER_MAP.get(backend_method, 1)
        _kwargs.pop("method", None)
        return _map_coordinates_xarray(src, trg, **_kwargs)
    elif method in ["nearest", "inverse_distance", "ordinary_kriging"]:
        if not (
            trg.attrs.get("source", None) == "wradlib"
            and trg.attrs.get("model", None) in ["kdtree"]
        ):
            trg = get_mapping(src, trg, **_kwargs)
        return _interpolate_mapping(src, trg, **_kwargs)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


@singledispatch
def interpolate_polar(data, *, mask=None, ipclass=Nearest):
    """
    Convenience function to interpolate polar data

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        2-dimensional array (azimuth, ranges) of floats;

        if no mask is assigned explicitly polar data should be a masked array
    mask : :class:`numpy:numpy.ndarray`, optional
        boolean array with pixels to be interpolated set to True,
        must have the same shape as data, defaults to None.
    ipclass : :class:`wradlib.ipol.IpolBase`
        A class which inherits from IpolBase, defaults to :class:`wradlib.ipol.Nearest`.

    Returns
    -------
    filled_data : :class:`numpy:numpy.ndarray`
        2D array with interpolated values for the values set to True in the mask

    Examples
    --------
    >>> import numpy as np  # noqa
    >>> import wradlib as wrl
    >>> # creating a data array and mask some values
    >>> data = np.arange(12.).reshape(4,3)
    >>> masked_values = (data==2) | (data==9)
    >>> # interpolate the masked data based on ''masked_values''
    >>> filled_a = wrl.ipol.interpolate_polar(data, mask = masked_values, ipclass = wrl.ipol.Linear)  # noqa
    >>> da = wrl.georef.create_xarray_dataarray(filled_a)
    >>> da = da.wrl.georef.georeference()
    >>> pm = wrl.vis.plot(da)
    >>> # the same result can be achieved by using a masked array instead of an explicit mask  # noqa
    >>> mdata = np.ma.array(data, mask = masked_values)
    >>> filled_b = wrl.ipol.interpolate_polar(mdata, ipclass = wrl.ipol.Linear)  # noqa
    >>> da = wrl.georef.create_xarray_dataarray(filled_b)
    >>> da = da.wrl.georef.georeference()
    >>> pm = wrl.vis.plot(da)


    """
    if mask is None:
        # no mask assigned: try to get it from masked array
        if not isinstance(data, np.ma.core.MaskedArray):
            util.warn(
                "Neither an explicit mask is assigned nor the data-array is masked."
            )
        mask = np.ma.getmaskarray(data)
    elif not np.any(mask):
        # mask contains no True values, so there is nothing to fill
        return data
    clutter_indices = np.where(mask.ravel())
    # construct the ranges for every bin
    ranges = np.tile(np.arange(0.5, data.shape[1] + 0.5), data.shape[0])
    # construct the angles for every bin
    angles = np.repeat(
        np.radians(np.linspace(0, 360, endpoint=False, num=data.shape[0])),
        data.shape[1],
    )
    # calculate cartesian coordinates for every bin
    binx = np.cos(angles) * ranges
    biny = np.sin(angles) * ranges
    # calculate cartesian coordinates for bins, which are not masked
    src_coord = np.array(
        [(np.delete(binx, clutter_indices)), (np.delete(biny, clutter_indices))]
    ).transpose()
    # calculate cartesian coordinates for bins, which are masked
    trg_coord = np.array([binx[clutter_indices], biny[clutter_indices]]).transpose()
    # data values for bins, which are not masked
    values_list = np.delete(data, clutter_indices)
    filled_data = data.copy().ravel()
    # interpolate masked bins
    filling = interpolate(src_coord, trg_coord, values_list, ipclass)
    # fill data with the interpolations
    filled_data[clutter_indices] = filling.astype(filled_data.dtype)
    # in case of nans as processed at the rim when interpolating linear,
    # these values are finally interpolated by nearest Neighbor interpolation
    if np.any(np.isnan(filled_data)):
        trg_coord = np.array(
            [
                binx[np.where(np.isnan(filled_data))],
                biny[np.where(np.isnan(filled_data))],
            ]
        ).transpose()
        filling = interpolate(src_coord, trg_coord, values_list, ipclass=Nearest)
        filled_data[np.where(np.isnan(filled_data))] = filling
    return filled_data.reshape(data.shape[0], data.shape[1])


@interpolate_polar.register(xr.DataArray)
def _interpolate_polar_xarray(obj, mask, **kwargs):
    dim0 = obj.wrl.util.dim0()

    def wrapper(obj, mask, *args, **kwargs):
        kwargs.setdefault("mask", mask)
        return interpolate_polar(obj, *args, **kwargs)

    out = xr.apply_ufunc(
        wrapper,
        obj,
        mask,
        input_core_dims=[[dim0, "range"], [dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        vectorize=True,
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "interpolate_polar"
    return out


def cart_to_irregular_interp(cartgrid, values, newgrid, **kwargs):
    util.warn(
        "`cart_to_irregular_interp` is deprecated; use `griddata` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return griddata(cartgrid, values, newgrid, **kwargs)


def griddata(cartgrid, values, newgrid, **kwargs):
    """
    Interpolate array ``values`` defined by cartesian coordinate array
    ``cartgrid`` to new coordinates defined by ``newgrid`` using
    nearest neighbour, linear or cubic interpolation

    Slow for large arrays

    Keyword arguments are fed to :func:`scipy:scipy.interpolate.griddata`

    Parameters
    ----------
    cartgrid : :class:`numpy:numpy.ndarray`
        3-dimensional array (nx, ny, lon/lat) of floats;
    values : :class:`numpy:numpy.ndarray`
        2-dimensional array (nx, ny) of data values
    newgrid : :class:`numpy:numpy.ndarray`
        Nx2-dimensional array (..., lon/lat) of floats
    kwargs : :func:`scipy:scipy.interpolate.griddata`

    Returns
    -------
    interp : :class:`numpy:numpy.ndarray`
        array with interpolated values of size N
    """

    # TODO: dimension checking

    newshape = newgrid.shape[:-1]

    cart_arr = cartgrid.reshape(-1, cartgrid.shape[-1])
    new_arr = newgrid.reshape(-1, newgrid.shape[-1])

    if values.ndim > 1:
        values = values.ravel()

    interp = sinterp.griddata(cart_arr, values, new_arr, **kwargs)
    interp = interp.reshape(newshape)

    return interp


def _griddata_xarray(src, trg, **kwargs):
    """Interpolate src DataArray onto irregular trg coordinates using scipy.griddata.

    Parameters
    ----------
    src : xr.DataArray
        Source 2D data (y, x) or with extra dims.
    trg : xr.DataArray
        Target coordinates (must have 'y' and 'x').
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic'.
    kwargs : dict
        Additional kwargs passed to griddata (e.g., fill_value).
    """
    kwargs.setdefault("fill_value", np.nan)

    def _griddata_2d(src_x, src_y, values, trg_x, trg_y, **kwargs):
        return sinterp.griddata((src_x, src_y), values, (trg_x, trg_y), **kwargs)

    order = dict(nearest=0, linear=1, cubic=3)
    kwargs.setdefault("method", "linear")

    src = util.crop(src, trg, pad=order[kwargs.get("method")])

    src_coords = kwargs.pop("src_coords")
    trg_coords = kwargs.pop("trg_coords")

    src2 = src.stack(points1=src.dims)
    trg2 = trg.stack(points2=trg.dims)

    out = xr.apply_ufunc(
        _griddata_2d,
        src2[src_coords.x],
        src2[src_coords.y],
        src2,
        trg2[trg_coords.x],
        trg2[trg_coords.y],
        input_core_dims=[
            ["points1"],
            ["points1"],
            ["points1"],
            ["points2"],
            ["points2"],
        ],
        output_core_dims=[["points2"]],
        vectorize=False,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=[src.dtype],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )

    return (
        out.unstack("points2")
        .transpose(*trg.dims)
        .assign_coords(trg.coords)
        .rename(src.name)
        .assign_attrs(src.attrs)
    )


def _cartesian_index(x_trg, y_trg, x_src, y_src, *, origin):
    nx = x_src.size
    ny = y_src.size

    cxmin = x_src.min()
    cxmax = x_src.max()
    cymin = y_src.min()
    cymax = y_src.max()

    xi = (nx - 1) * (x_trg - cxmin) / (cxmax - cxmin)

    if origin == "lower":
        yi = (ny - 1) * (y_trg - cymin) / (cymax - cymin)
    else:
        yi = ny - (ny - 1) * (y_trg - cymin) / (cymax - cymin)

    return xi, yi


@singledispatch
def _cartesian_to_indices(cartgrid, newgrid, *, origin="lower"):
    """
    Compute floating-point indices into ``values`` for spline interpolation.

    Parameters
    ----------
    cartgrid : ndarray
        (ny, nx, 2) array defining cartesian grid (x/y or lon/lat)
    newgrid : ndarray
        (..., 2) array defining target coordinates
    origin : {"lower", "upper"}

    Returns
    -------
    xi, yi : ndarray
        Floating-point indices with shape newgrid.shape[:-1]
    """
    x_src = cartgrid[0, ..., 0]
    y_src = cartgrid[..., 0, 1]

    x_trg = newgrid[..., 0].ravel()
    y_trg = newgrid[..., 1].ravel()

    xi, yi = _cartesian_index(
        x_trg,
        y_trg,
        x_src,
        y_src,
        origin=origin,
    )

    return xi, yi


@_cartesian_to_indices.register(xr.DataArray)
def _cartesian_to_indices_xarray(src, trg, **kwargs):
    """xarray-native precomputation of spline indices."""
    src_coords = kwargs.pop("src_coords")
    trg_coords = kwargs.pop("trg_coords")

    sx = src_coords.x
    sy = src_coords.y
    tx = trg_coords.x
    ty = trg_coords.y

    x_src = src.coords[sx]
    y_src = src.coords[sy]
    origin = "lower" if y_src[1] > y_src[0] else "upper"

    x_trg = trg.coords[tx].stack(points=trg.coords[tx].dims)
    y_trg = trg.coords[ty].stack(points=trg.coords[ty].dims)

    xi, yi = _cartesian_index(
        x_trg,
        y_trg,
        x_src,
        y_src,
        origin=origin,
    )

    return xr.Dataset(
        {"xi": xi, "yi": yi},
        coords={"points": x_trg.points},
    )


def cart_to_irregular_spline(cartgrid, values, newgrid, **kwargs):
    util.warn(
        "`cart_to_irregular_spline` is deprecated; use `map_coordinates` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return map_coordinates(cartgrid, values, newgrid, **kwargs)


def map_coordinates(cartgrid, values, newgrid, **kwargs):
    """
    Map array ``values`` defined by cartesian coordinate array ``cartgrid``
    to new coordinates defined by ``newgrid`` using spline interpolation.

    Keyword arguments are fed through to
    :func:`scipy:scipy.ndimage.map_coordinates`

    Parameters
    ----------
    cartgrid : :class:`numpy:numpy.ndarray`
        3-dimensional array (nx, ny, lon/lat) of floats
    values : :class:`numpy:numpy.ndarray`
        2-dimensional array (nx, ny) of data values
    newgrid : :class:`numpy:numpy.ndarray`
        Nx2-dimensional array (..., lon/lat) of floats
    kwargs : :func:`scipy:scipy.ndimage.map_coordinates`

    Returns
    -------
    interp : :class:`numpy:numpy.ndarray`
        array with interpolated values of size N

    Examples
    --------
    See :ref:`notebooks:notebooks/beamblockage/beamblockage:preprocessing the dem`.
    """
    kwargs.setdefault("mode", "nearest")
    kwargs.setdefault("cval", np.nan)
    kwargs.setdefault("order", 1)

    xi, yi = _cartesian_to_indices(
        cartgrid,
        newgrid,
        origin=util.get_raster_origin(cartgrid),
    )

    interp = ndimage.map_coordinates(values, [yi, xi], **kwargs)
    return interp.reshape(newgrid.shape[:-1])


def _map_coordinates_xarray(src, trg, **kwargs):
    """
    Apply spline interpolation to an xarray DataArray.
    """
    kwargs.setdefault("mode", "nearest")
    kwargs.setdefault("cval", np.nan)
    kwargs.setdefault("order", 1)

    src = util.crop(src, trg, pad=kwargs.get("order"))

    indices = _cartesian_to_indices(src, trg, **kwargs)

    src_coords = kwargs.get("src_coords")
    sx = src_coords.x
    sy = src_coords.y

    kwargs.pop("src_coords")
    kwargs.pop("trg_coords")

    def _map_coordinates_2d(arr, yi, xi, **kwargs):
        coords = np.vstack([yi, xi])
        return ndimage.map_coordinates(arr, coords, **kwargs)

    out = xr.apply_ufunc(
        _map_coordinates_2d,
        src,
        indices["yi"],
        indices["xi"],
        input_core_dims=[[sy, sx], ["points"], ["points"]],
        output_core_dims=[["points"]],
        vectorize=False,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=[src.dtype],
    )

    return (
        out.unstack("points")
        .transpose(*trg.dims)
        .assign_coords(trg.coords)
        .rename(src.name)
        .assign_attrs(src.attrs)
    )


class IpolMethods(XarrayMethods):
    """wradlib xarray SubAccessor methods for Ipol Methods."""

    @docstring(interpolate_polar)
    def interpolate_polar(self, *args, **kwargs):
        if not isinstance(self, IpolMethods):
            return interpolate_polar(self, *args, **kwargs)
        else:
            return interpolate_polar(self._obj, *args, **kwargs)

    @docstring(get_mapping)
    def get_mapping(self, *args, **kwargs):
        if not isinstance(self, IpolMethods):
            return get_mapping(self, *args, **kwargs)
        else:
            return get_mapping(self._obj, *args, **kwargs)

    @docstring(_interpolate_xarray)
    def interpolate(self, *args, **kwargs):
        if not isinstance(self, IpolMethods):
            return interpolate(self, *args, **kwargs)
        else:
            return interpolate(self._obj, *args, **kwargs)


if __name__ == "__main__":
    print("wradlib: Calling module <ipol> as main...")
