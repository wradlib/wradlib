#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raster Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "read_gdal_values",
    "read_gdal_projection",
    "read_gdal_coordinates",
    "extract_raster_dataset",
    "get_raster_extent",
    "get_raster_elevation",
    "reproject_raster_dataset",
    "merge_raster_datasets",
    "create_raster_dataset",
    "set_raster_origin",
    "set_raster_indexing",
    "set_coordinate_indexing",
    "raster_to_polyvert",
    "create_raster_geographic",
    "create_raster_xarray",
]
__doc__ = __doc__.format("\n   ".join(__all__))


import numpy as np
import xarray as xr

import wradlib
from wradlib import georef
from wradlib.util import import_optional, warn

gdal = import_optional("osgeo.gdal")
gdal_array = import_optional("osgeo.gdal_array")
osr = import_optional("osgeo.osr")


def _pixel_coordinates(nx, ny, mode):
    """Get the pixel coordinates of an image.

    Parameters
    ----------
    nx : int
        x size (number of columns)
    ny : int
        y size (numbers or rows)
    mode : str
        either 'center' (0.5 1.5 ...) or 'edge' (0 1 ...)

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
         array containing pixel coordinates (x,y) in image convention
         shape is (nrows, ncols, 2) if mode==center
         shape is (nrows+1, ncols+1, 2) if mode==edge

    """
    if mode == "center":
        x = np.linspace(0.5, nx - 0.5, num=nx)
        y = np.linspace(0.5, ny - 0.5, num=ny)

    if mode == "edge":
        x = np.linspace(0, nx, num=nx + 1)
        y = np.linspace(0, ny, num=ny + 1)

    X, Y = np.meshgrid(x, y)
    coordinates = np.stack((X, Y), axis=-1)

    return coordinates


def _pixel_to_map(coordinates, geotransform):
    """Apply a geographical transformation to return map coordinates from
    pixel coordinates.

    Parameters
    ----------
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector:

            - geotransform[0] = East/West location of Upper Left corner
            - geotransform[1] = X pixel size
            - geotransform[2] = X pixel rotation
            - geotransform[3] = North/South location of Upper Left corner
            - geotransform[4] = Y pixel rotation
            - geotransform[5] = Y pixel size

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        2d array with map coordinates (x,y)
    """
    coordinates_map = np.empty(coordinates.shape)
    coordinates_map[..., 0] = (
        geotransform[0]
        + geotransform[1] * coordinates[..., 0]
        + geotransform[2] * coordinates[..., 1]
    )
    coordinates_map[..., 1] = (
        geotransform[3]
        + geotransform[4] * coordinates[..., 0]
        + geotransform[5] * coordinates[..., 1]
    )
    return coordinates_map


def read_gdal_coordinates(dataset, *, mode="center"):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing
    mode : str
        either 'center' or 'edge'

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
        Array of shape (nrows,ncols,2) containing xy coordinates.
        The array indexing follows image convention with origin
        at upper left pixel.
        The shape is (nrows+1,ncols+1,2) if mode == edge.

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    coordinates_pixel = _pixel_coordinates(
        dataset.RasterXSize, dataset.RasterYSize, mode
    )

    geotransform = dataset.GetGeoTransform()
    coordinates = _pixel_to_map(coordinates_pixel, geotransform)

    return coordinates


def read_gdal_projection(dataset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        dataset projection object

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    wkt = dataset.GetProjection()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(wkt)
    return crs


def read_gdal_values(dataset, *, nodata=None):
    """Read values from a gdal object.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing
    nodata : float
        replace nodata values

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols) or (nbands, nrows, ncols)
        containing the data values.

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_cloud.ipynb`.

    """
    nbands = dataset.RasterCount

    # data values
    bands = []
    for i in range(nbands):
        band = dataset.GetRasterBand(i + 1)
        nd = band.GetNoDataValue()
        data = band.ReadAsArray()
        if nodata is not None:
            data[data == nd] = nodata
        bands.append(data)

    return np.squeeze(np.array(bands))


def extract_raster_dataset(dataset, *, mode="center", nodata=None):
    """Extract data, coordinates and projection information

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster dataset
    mode : str
        either 'center' or 'edge'
    nodata : float
        replace nodata values

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols) or (nbands, nrows, ncols)
        containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrows,ncols,2) containing xy coordinates.
        The array indexing follows image convention with origin
        at the upper left pixel (northup).
        The shape is (nrows+1,ncols+1,2) if mode == edge.
    projection : :py:class:`gdal:osgeo.osr.SpatialReference`
        Spatial reference system of the used coordinates.
    """

    values = read_gdal_values(dataset, nodata=nodata)

    coords = read_gdal_coordinates(dataset, mode=mode)

    projection = read_gdal_projection(dataset)

    return values, coords, projection


def get_raster_extent(dataset, *, geo=False, window=True):
    """Get the coordinates of the 4 corners of the raster dataset

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)
    geo : bool
        True to get geographical coordinates
    window : bool
        True to get the window containing the corners

    Returns
    -------
    extent : :class:`numpy:numpy.ndarray`
        corner coordinates [ul,ll,lr,ur] or
        window extent [xmin, xmax, ymin, ymax]
    """

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    geotrans = dataset.GetGeoTransform()
    xmin = geotrans[0]
    ymax = geotrans[3]
    xmax = geotrans[0] + geotrans[1] * x_size
    ymin = geotrans[3] + geotrans[5] * y_size

    extent = np.array([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]])

    if geo:
        crs = read_gdal_projection(dataset)
        extent = georef.reproject(extent, src_crs=crs)

    if window:
        x = extent[:, 0]
        y = extent[:, 1]
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

    return extent


def get_raster_elevation(dataset, *, resample=None, **kwargs):
    """Return surface elevation corresponding to raster dataset
       The resampling algorithm is chosen based on scale ratio

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)
    resample : :py:class:`gdal:osgeo.gdalconst.ResampleAlg`
        If None the best algorithm is chosen based on scales.
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    kwargs : dict
        keyword arguments passed to :func:`wradlib.io.dem.get_srtm`

    Returns
    -------
    elevation : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing elevation
    """
    extent = get_raster_extent(dataset)
    src_ds = wradlib.io.dem.get_srtm(extent, **kwargs)

    driver = gdal.GetDriverByName("MEM")
    dst_ds = driver.CreateCopy("ds", dataset)

    if resample is None:
        src_gt = src_ds.GetGeoTransform()
        dst_gt = dst_ds.GetGeoTransform()
        src_scale = min(abs(src_gt[1]), abs(src_gt[5]))
        dst_scale = min(abs(dst_gt[1]), abs(dst_gt[5]))
        ratio = dst_scale / src_scale

        resample = gdal.GRA_Bilinear
        if ratio > 2:
            resample = gdal.GRA_Average
        if ratio < 0.5:
            resample = gdal.GRA_NearestNeighbour

    gdal.Warp(
        dst_ds,
        src_ds,
        dstSRS=dst_ds.GetProjection(),
        srcSRS=src_ds.GetProjection(),
        resampleAlg=resample,
    )
    elevation = read_gdal_values(dst_ds)

    return elevation


def set_raster_origin(data, coords, direction):
    """Converts Data and Coordinates Origin

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    direction : str
        'lower' or 'upper', direction in which to convert data and coordinates.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    """
    x_sp, y_sp = coords[1, 1] - coords[0, 0]
    origin = "lower" if y_sp > 0 else "upper"
    same = origin == direction
    if not same:
        data = np.flip(data, axis=-2)
        coords = np.flip(coords, axis=-3)

    return data, coords


def set_raster_indexing(data, coords, *, indexing="xy"):
    """Sets Data and Coordinates Indexing Scheme

    This converts data and coordinate layout from row-major to column major indexing.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N) containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N, 2) containing xy-coordinates.
    indexing : str
        'xy' or 'ij', indexing scheme in which to convert data and coordinates.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M) containing the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M, 2) containing xy-coordinates.
    """
    shape = coords.shape[:-1]

    if shape != data.shape:
        raise ValueError(
            f"coordinate shape {coords.shape} and data shape " f"{data.shape} mismatch."
        )

    coords = set_coordinate_indexing(coords, indexing=indexing)

    # if coordinate shape has changed, we need to transform data too
    if coords.shape[:-1] != shape:
        data_shape = tuple(range(data.ndim - 2)) + (-1, -2)
        data = data.transpose(data_shape)

    return data, coords


def set_coordinate_indexing(coords, *, indexing="xy"):
    """Sets Coordinates Indexing Scheme

    This converts coordinate layout from row-major to column major indexing.

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., M, N, 2) containing xy-coordinates.
    indexing : str
        'xy' or 'ij', indexing scheme in which to convert data and coordinates.

    Returns
    -------
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (..., N, M, 2) containing xy-coordinates.
    """
    is_grid = hasattr(coords, "shape") and coords.ndim >= 3 and coords.shape[-1] == 2
    if not is_grid:
        raise ValueError(
            f"wrong coordinate shape {coords.shape}, " f"(..., M, N, 2) expected."
        )
    if indexing not in ["xy", "ij"]:
        raise ValueError(f"Unknown indexing value {indexing}. Use either `xy` or `ij`.")

    rowcol = coords[0, 0, 1] == coords[0, 1, 1]
    convert = (rowcol and indexing == "ij") or (not rowcol and indexing == "xy")

    if convert:
        coords_shape = tuple(range(coords.ndim - 3)) + (-2, -3, -1)
        coords = coords.transpose(coords_shape)

    return coords


def reproject_raster_dataset(src_ds, **kwargs):
    """Reproject/Resample given dataset according to keyword arguments

    Parameters
    ----------
    src_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)

    Keyword Arguments
    -----------------
    spacing : float
        float or tuple of two floats
        pixel spacing of destination dataset, same unit as pixel coordinates
    size : int
        tuple of two ints
        X/YRasterSize of destination dataset
    resample : :py:class:`gdal:osgeo.gdalconst.ResampleAlg`
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    projection_source : :py:class:`gdal:osgeo.osr.SpatialReference`
        source dataset projection, defaults to None (get projection from src_ds)
    projection_target : :py:class:`gdal:osgeo.osr.SpatialReference`
        destination dataset projection, defaults to None
    align : bool or tuple
        If False, there is no destination grid aligment.
        If True, aligns the destination grid to the next integer multiple of
        destination grid.
        If tuple (upper-left x,y-coordinate), the destination grid is aligned to this point.

    Returns
    -------
    dst_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        reprojected/resampled raster dataset
    """

    # checking kwargs
    spacing = kwargs.pop("spacing", None)
    size = kwargs.pop("size", None)
    resample = kwargs.pop("resample", gdal.GRA_Bilinear)
    src_crs = kwargs.pop("src_crs", None)
    trg_crs = kwargs.pop("trg_crs", None)
    align = kwargs.pop("align", False)

    if spacing is None and size is None:
        raise NameError("Either keyword `spacing` or `size` must be given.")

    if spacing is not None and size is not None:
        warn("Both `spacing` and `size` kwargs given, `size` will be ignored.")

    # Get the GeoTransform vector
    src_geo = src_ds.GetGeoTransform()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    # get extent
    ulx = src_geo[0]
    uly = src_geo[3]
    lrx = src_geo[0] + src_geo[1] * x_size
    lry = src_geo[3] + src_geo[5] * y_size

    extent = np.array([[[ulx, uly], [lrx, uly]], [[ulx, lry], [lrx, lry]]])

    if trg_crs:
        # try to load projection from source dataset if None is given
        if src_crs is None:
            src_proj = src_ds.GetProjection()
            if not src_proj:
                raise ValueError(
                    "`src_ds` is missing projection information, please use `src_crs`-kwarg "
                    "and provide a fitting GDAL OSR SRS object."
                )
            src_crs = osr.SpatialReference()
            src_crs.ImportFromWkt(src_proj)

        # Transformation
        extent = georef.reproject(extent, src_crs=src_crs, trg_crs=trg_crs)

        # wkt needed
        src_crs = src_crs.ExportToWkt()
        trg_crs = trg_crs.ExportToWkt()

    (ulx, uly, urx, ury, llx, lly, lrx, lry) = tuple(list(extent.flatten().tolist()))

    # align grid to destination raster or UL-corner point
    if align:
        try:
            ulx, uly = align
        except TypeError:
            pass

        ulx = int(max(np.floor(ulx), np.floor(llx)))
        uly = int(min(np.ceil(uly), np.ceil(ury)))
        lrx = int(min(np.ceil(lrx), np.ceil(urx)))
        lry = int(max(np.floor(lry), np.floor(lly)))

    # calculate cols/rows or xspacing/yspacing
    if spacing:
        try:
            x_ps, y_ps = spacing
        except TypeError:
            x_ps = spacing
            y_ps = spacing

        cols = int(abs(lrx - ulx) / x_ps)
        rows = int(abs(uly - lry) / y_ps)
    elif size:
        cols, rows = size
        x_ps = x_size * src_geo[1] / cols
        y_ps = y_size * abs(src_geo[5]) / rows

    # create destination in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")

    # and set RasterSize according ro cols/rows
    dst_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Float32)

    # Create the destination GeoTransform with changed x/y spacing
    dst_geo = (ulx, x_ps, src_geo[2], uly, src_geo[4], -y_ps)

    # apply GeoTransform to destination dataset
    dst_ds.SetGeoTransform(dst_geo)

    # apply Projection to destination dataset
    if trg_crs is not None:
        dst_ds.SetProjection(trg_crs)
    else:
        dst_ds.SetProjection(src_ds.GetProjection())

    # nodata handling, need to initialize dst_ds with nodata
    src_band = src_ds.GetRasterBand(1)
    nodata = src_band.GetNoDataValue()
    dst_band = dst_ds.GetRasterBand(1)
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)
        dst_band.WriteArray(np.ones((rows, cols)) * nodata)
    dst_band.FlushCache()

    # resample and reproject dataset
    gdal.Warp(
        dst_ds,
        src_ds,
        dstSRS=trg_crs,
        srcSRS=src_crs,
        resampleAlg=resample,
    )
    return dst_ds


def create_raster_dataset(data, coords, *, crs=None, nodata=-9999):
    """Create In-Memory Raster Dataset

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (nrows, ncols, 2) containing pixel center coordinates
        or
        Array of shape (nrows+1, ncols+1, 2) containing pixel edge coordinates
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Spatial reference system of the used coordinates, defaults to None.
    nodata : int
        Value of NODATA

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        In-Memory raster dataset

    Note
    ----
    The origin of the provided data and coordinates is UPPER LEFT.
    """

    # align data
    data = data.copy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, rows, cols = data.shape

    # create In-Memory Raster with correct dtype
    mem_drv = gdal.GetDriverByName("MEM")
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
    dataset = mem_drv.Create("", cols, rows, bands, gdal_type)

    # initialize geotransform
    x_ps, y_ps = coords[1, 1] - coords[0, 0]
    if data.shape[-2:] == coords.shape[0:2]:
        upper_corner_x = coords[0, 0, 0] - x_ps / 2
        upper_corner_y = coords[0, 0, 1] - y_ps / 2
    else:
        upper_corner_x = coords[0, 0, 0]
        upper_corner_y = coords[0, 0, 1]
    geotran = [upper_corner_x, x_ps, 0, upper_corner_y, 0, y_ps]
    dataset.SetGeoTransform(geotran)

    if crs:
        dataset.SetProjection(crs.ExportToWkt())

    # set np.nan to nodata
    dataset.GetRasterBand(1).SetNoDataValue(nodata)

    for i, band in enumerate(data, start=1):
        dataset.GetRasterBand(i).WriteArray(band)

    return dataset


def merge_raster_datasets(datasets, **kwargs):
    """Merge rasters.

    Parameters
    ----------
    datasets : list
        list of :py:class:`gdal:osgeo.gdal.Dataset`
        raster images with georeferencing
    kwargs : dict
        keyword arguments passed to gdal.Warp()

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        merged raster dataset
    """

    dataset = gdal.Warp("", datasets, format="MEM", **kwargs)

    return dataset


def raster_to_polyvert(dataset):
    """Get raster polygonal vertices from gdal dataset.

    Parameters
    ----------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        raster image with georeferencing (GeoTransform at least)

    Returns
    -------
    polyvert : :class:`numpy:numpy.ndarray`
        A N-d array of polygon vertices with shape (..., 5, 2).

    """
    rastercoords = read_gdal_coordinates(dataset, mode="edge")

    polyvert = georef.grid_to_polyvert(rastercoords)

    return polyvert


def create_raster_xarray(crs, bounds, size):
    """Create a georeferenced raster image using xarray model and gdal conventions.

    Parameters
    ----------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        coordinate reference system mapping geographic coordinates to x,y coordinates
    bounds : tuple of floats
        image bounds (x_min, x_max, y_min, y_max)
    size : float or tuple of floats
        (x,y) pixel size in meters (or degrees if crs is geographic)

    Returns
    -------
    raster : :class:`xarray:Dataarray`
        raster image dataset

    """

    xmin, xmax, ymin, ymax = bounds

    try:
        xsize, ysize = size
    except TypeError:
        xsize, ysize = size, size

    xmin = xmin - xmin % xsize
    ymin = ymin - ymin % ysize
    if xmax % xsize != 0:
        xmax = xmax - xmax % xsize + xsize
    if ymax % ysize != 0:
        ymax = ymax - ymax % ysize + ysize

    geotransform = [xmin, xsize, 0, ymax, 0, -ysize]
    geotransform = [str(r) for r in geotransform]
    geotransform = " ".join(geotransform)

    num = (xmax - xmin) / xsize + 1
    x = np.linspace(xmin, xmax, num=int(num), endpoint=True)
    num = (ymax - ymin) / ysize + 1
    y = np.linspace(ymin, ymax, num=int(num), endpoint=True)
    if x[-1] != xmax:
        x = np.append(x, xmax)
    if y[-1] != ymax:
        y = np.append(y, ymax)

    y = np.flip(y)

    raster = xr.DataArray(dims=["x", "y"], coords={"x": x, "y": y})

    wkt = crs.ExportToWkt()
    raster = raster.assign_coords({"spatial_ref": 0})
    raster.spatial_ref.attrs["crs_wkt"] = wkt
    raster.spatial_ref.attrs["GeoTransform"] = geotransform

    return raster


def create_raster_geographic(bounds, size, size_in_meters=False):
    """Create a geographic raster.

    Parameters
    ----------

    bounds : (lon_min, lon_max, lat_min, lat_max)
        geographic bounds
    size : int
        pixel size in degrees
    size_in_meters : bool
        if True, size is in meters and geographic sizes are approximated to keep bounds fixed

    Returns
    -------
    raster : :class:`xarray:Dataset`
        raster image dataset

    """
    crs = georef.get_earth_projection()

    if size_in_meters:
        xsize, ysize = georef.geographic_size(bounds, size)
        xmin, xmax, ymin, ymax = bounds
        lx = xmax - xmin
        n = round(lx / xsize)
        xsize = lx / n
        ly = ymax - ymin
        m = round(ly / ysize)
        ysize = ly / m
        size = (xsize, ysize)

    raster = georef.create_raster_xarray(crs, bounds, size)

    return raster
