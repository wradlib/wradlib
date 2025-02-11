#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Composition
^^^^^^^^^^^

Combine data from different radar locations on one common set of locations

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "extract_circle",
    "togrid",
    "compose_ko",
    "compose_weighted",
    "CompMethods",
    "create_grid",
    "sweep_to_raster",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from functools import singledispatch

import numpy as np
from xarray import DataArray, Dataset, apply_ufunc, broadcast, concat

from wradlib.georef import epsg_to_osr, reproject, wkt_to_osr
from wradlib.ipol import RectBin
from wradlib.util import XarrayMethods, docstring


def extract_circle(center, radius, coords):
    """Extract the indices of ``coords`` which fall within a circle \
    defined by ``center`` and ``radius``.

    Parameters
    ----------
    center : float
    radius : float
    coords : :class:`numpy:numpy.ndarray`
        array of float with shape (numpoints, 2)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        1-darray of integers, index array referring to the ``coords`` array

    """
    return np.where(((coords - center) ** 2).sum(axis=-1) < radius**2)[0]


@singledispatch
def togrid(src, trg, radius, center, data, interpol, *args, **kwargs):
    """Interpolate data from a radar location to the composite grid or set of \
    locations

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        array of float of shape (numpoints, ndim),
        cartesian x / y coordinates of the radar bins
    trg : :class:`numpy:numpy.ndarray`
        array of float of shape (numpoints, ndim),
        cartesian x / y coordinates of the composite
    radius : float
        the radius of the radar circle (same units as src and trg)
    center : :class:`numpy:numpy.ndarray`
        array of float, the location coordinates of the radar
    data : :class:`numpy:numpy.ndarray`
        array of float, the data that should be transferred to composite
    interpol : :class:`~wradlib.ipol.IpolBase`
        an interpolation class name from :mod:`wradlib.ipol`
        e.g. :class:`~wradlib.ipol.Nearest` or :class:`~wradlib.ipol.Idw`

    Other Parameters
    ----------------
    *args : dict
        arguments of Interpolator (see class documentation)

    Keyword Arguments
    -----------------
    **kwargs : dict
        keyword arguments of Interpolator (see class documentation)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        array of float, data of the radar circle which is interpolated on
        the composite grid

    Note
    ----
    Keyword arguments to be used while calling the interpolator can be issued as
    `call_kwargs`, e.g. togrid(..., call_kwargs=dict(maxdist=10))

    Examples
    --------

    See :ref:`/notebooks/basics/wradlib_workflow.ipynb#Gridding`.

    """
    # get indices to select the subgrid from the composite grid
    ix = extract_circle(center, radius, trg)
    call_kwargs = kwargs.pop("call_kwargs", {})
    # interpolate on subgrid
    ip = interpol(src, trg[ix], *args, **kwargs)
    data_on_subgrid = ip(data, **call_kwargs).reshape(len(ix))
    # create container for entire grid
    composegridshape = [len(trg)]
    composegridshape.extend(data.shape[1:])
    compose_grid = np.repeat(np.nan, len(trg) * np.prod(data.shape[1:])).reshape(
        composegridshape
    )
    # push subgrid results into the large grid
    compose_grid[ix] = data_on_subgrid
    return compose_grid


@togrid.register(DataArray)
def _togrid_xarray(obj, trg, *args, **kwargs):
    dim0 = obj.wrl.util.dim0()
    grid_xy = (
        concat(broadcast(trg.y, trg.x), "xy")
        .stack(npoints_cart=("y", "x"))
        .transpose(..., "xy")
    )
    xy = (
        concat([obj.y, obj.x], "xy")
        .stack(npoints_pol=(dim0, "range"))
        .transpose(..., "xy")
        .reset_coords(drop=True)
    )
    obj = obj.stack(npoints_pol=(dim0, "range")).reset_coords(drop=True)

    def wrapper(obj, xy, grid_xy, **kwargs):
        radius = kwargs.get("radius")
        center = kwargs.get("center")
        ipol = kwargs.get("interpol")
        out = togrid(xy, grid_xy, radius, center, obj, ipol)
        return out

    out = apply_ufunc(
        wrapper,
        obj,
        xy,
        grid_xy,
        input_core_dims=[
            ["npoints_pol"],
            ["npoints_pol", "xy"],
            ["npoints_cart", "xy"],
        ],
        output_core_dims=[["npoints_cart"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out = out.unstack("npoints_cart")
    out.attrs = obj.attrs
    out.name = f"{obj.name}.togrid"
    return out


def compose_ko(radargrids, qualitygrids):
    """Composes grids according to quality information using quality \
    information as a knockout criterion.

    The value of the composed pixel is taken from the radargrid whose
    quality grid has the highest value.

    Parameters
    ----------
    radargrids : list
        radar data to be composited. Each item in the list corresponds to the
        data of one radar location. All items must have the same shape.
    qualitygrids : list
        quality data to decide upon which radar site will contribute its pixel
        to the composite. Then length of this list must be the same as that
        of `radargrids`. All items must have the same shape and be aligned with
        the items in `radargrids`.


    Returns
    -------
    composite : :class:`numpy:numpy.ndarray`

    """
    # first add a fallback array for all pixels having missing values in all
    # radargrids
    radarfallback = np.repeat(np.nan, np.prod(radargrids[0].shape)).reshape(
        radargrids[0].shape
    )
    radargrids.append(radarfallback)
    radarinfo = np.array(radargrids)
    # then do the same for the quality grids
    qualityfallback = np.repeat(-np.inf, np.prod(radargrids[0].shape)).reshape(
        radargrids[0].shape
    )
    qualitygrids.append(qualityfallback)
    qualityinfo = np.array(qualitygrids)

    select = np.nanargmax(qualityinfo, axis=0)
    composite = radarinfo.reshape((radarinfo.shape[0], -1))[
        select.ravel(), np.arange(np.prod(radarinfo.shape[1:]))
    ].reshape(radarinfo.shape[1:])
    radargrids.pop()
    qualitygrids.pop()

    return composite


@singledispatch
def compose_weighted(radargrids, qualitygrids):
    """Composes grids according to quality information using a weighted \
    averaging approach.

    The value of the composed pixel is the weighted average of all radar
    pixels with the quality values being the weights.

    Parameters
    ----------
    radargrids : list
        list of arrays
    qualitygrids : list
        list of arrays

    Returns
    -------
    composite : :class:`numpy:numpy.ndarray`

    Examples
    --------

    See :ref:`/notebooks/workflow/recipe1.ipynb`.

    See Also
    --------
    :func:`~wradlib.comp.compose_ko`
    """
    radarinfo = np.array(radargrids)
    qualityinfo = np.array(qualitygrids)
    # overall nanmask
    nanmask = np.all(np.isnan(radarinfo), axis=0)
    # quality grids must contain values only where radarinfo does
    qualityinfo[np.isnan(radarinfo)] = np.nan

    qualityinfo /= np.nansum(qualityinfo, axis=0)

    composite = np.nansum(radarinfo * qualityinfo, axis=0)
    composite[nanmask] = np.nan

    return composite


@compose_weighted.register(DataArray)
def _compose_weighted_xarray(radargrids, qualitygrids):
    qualityinfo = qualitygrids.where(radargrids)
    qualityinfo /= qualityinfo.sum("radar", skipna=True)
    composite = (radargrids * qualityinfo).sum("radar", skipna=True)
    composite.name = radargrids.name
    composite.attrs = radargrids.attrs
    return composite.where(radargrids.sum("radar"))


class CompMethods(XarrayMethods):
    """wradlib xarray SubAccessor methods for Ipol Methods."""

    @docstring(togrid)
    def togrid(self, *args, **kwargs):
        if not isinstance(self, CompMethods):
            return togrid(self, *args, **kwargs)
        else:
            return togrid(self._obj, *args, **kwargs)

    @docstring(compose_weighted)
    def compose_weighted(self, *args, **kwargs):
        if not isinstance(self, CompMethods):
            return compose_weighted(self, *args, **kwargs)
        else:
            return compose_weighted(self._obj, *args, **kwargs)


def create_grid(bounds, size, crs=None, lonlat=False, raster=False, georef=False):
    """Create a grid for compositing radar data

    Parameters
    ----------
    bounds : tuple
        limits of the grid
        x_min, y_min, x_max, y_max
        lon_min, lat_min, lon_max, lat_max (if crs is None)
    size : float
        grid size in meters (or degress if crs is None)
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        coordinate reference system used for the projection
    lonlat : boolean
        True to provide bounds as longitude and latitude if crs is not None
    raster : boolean
        True to create a raster image with pixel center coordinates and north up
    georef : boolean
        True to add georeference metadata

    Returns
    -------
    grid : :class:`xarray:Dataset`
        composite grid

    """
    if crs is None:
        crs = epsg_to_osr(4326)

    if lonlat:
        lon_min, lat_min, lon_max, lat_max = bounds
        lon_mid = lon_min / 2 + lon_max / 2
        lat_mid = lat_min / 2 + lat_max / 2
        (xmin, temp) = reproject((lon_min, lat_mid), trg_crs=crs)
        (temp, ymin) = reproject((lon_mid, lat_min), trg_crs=crs)
        (xmax, temp) = reproject((lon_max, lat_mid), trg_crs=crs)
        (temp, ymax) = reproject((lon_mid, lat_max), trg_crs=crs)
    else:
        xmin, ymin, xmax, ymax = bounds

    xmin = xmin - xmin % size
    ymin = ymin - ymin % size
    if xmax % size != 0:
        xmax = xmax - xmax % size + size
    if ymax % size != 0:
        ymax = ymax - ymax % size + size

    if georef:
        geotransform = [xmin, size, 0, ymax, 0, -size]
        geotransform = [str(r) for r in geotransform]
        geotransform = " ".join(geotransform)

    if raster:
        xmin = xmin + size / 2
        ymin = ymin + size / 2
        xmax = xmax - size / 2
        ymax = ymax - size / 2

    num = (xmax - xmin) / size + 1
    x = np.linspace(xmin, xmax, num=int(num), endpoint=True)
    num = (ymax - ymin) / size + 1
    y = np.linspace(ymin, ymax, num=int(num), endpoint=True)
    if x[-1] != xmax:
        x = np.append(x, xmax)
    if y[-1] != ymax:
        y = np.append(y, ymax)

    if raster:
        y = np.flip(y)

    grid = Dataset(coords={"x": (["x"], x), "y": (["y"], y)})

    if georef:
        wkt = crs.ExportToWkt()
        grid = grid.assign({"spatial_ref": 0})
        grid.spatial_ref.attrs["crs_wkt"] = wkt
        grid.spatial_ref.attrs["GeoTransform"] = geotransform

    return grid


def sweep_to_raster(sweep, raster, transform=None, reuse=False):
    """Transform a radar sweep into a raster image.

    Parameters
    ----------
    sweep : :class:`xarray:xarray.dataset`
        radar sweep dataset following WMO conventions
    raster : :class:`xarray:xarray.dataset`
        raster image dataset with north up
    transform : :class:`wradlib.ipol.RectBin`
        the transformation object from a previous call
    reuse : boolean
        True to return the transformation object

    Returns
    -------
    raster : :class:`xarray:dataset`
        raster dataset with combined sweep values
    transform : :class:`wradlib.ipol.RectBin`
        the transformation object for reuse

    """
    if transform is None:
        wkt = raster.spatial_ref.attrs["crs_wkt"]
        crs = wkt_to_osr(wkt)
        sweep = sweep.wrl.georef.georeference(crs=crs)
        coord_sweep = np.dstack((sweep.x, sweep.y))
        x, y = np.meshgrid(raster.x, raster.y)
        coord_raster = np.dstack((x, y))

        radius = sweep.range.values[-1]
        lon = float(sweep.longitude.values)
        lat = float(sweep.latitude.values)
        fill = extract_circle((lon, lat), radius, coord_raster)
        transform = RectBin(coord_sweep, coord_raster, fill=fill)

    raster = raster.copy()
    for varname in sweep.data_vars:
        if sweep[varname].ndim == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = transform(sweep[varname].values, statistic=np.nanmean)
        raster[varname] = (("y", "x"), val)
        raster[varname].attrs = sweep[varname].attrs
        raster[varname].attrs["grid_mapping"] = "spatial_ref"

    if reuse:
        return raster, transform
    else:
        return raster


if __name__ == "__main__":
    print("wradlib: Calling module <comp> as main...")
