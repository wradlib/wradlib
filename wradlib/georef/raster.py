#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raster Functions
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_gdal_values
   read_gdal_projection
   read_gdal_coordinates
   extract_raster_dataset
   reproject_raster_dataset
   create_raster_dataset
   set_raster_origin
"""
import numpy as np
from osgeo import gdal, osr, gdal_array

from wradlib import georef, util


def _pixel_coordinates(nx, ny, mode):
    """Get the pixel coordinates of an image.

    Parameters
    ----------
    nx : int
        x size (number of columns)
    ny : int
        y size (numbers or rows)
    mode : string
        either 'centers' (0.5 1.5 ...) or 'edges' (0 1 ...)

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
         array containing pixel coordinates (x,y) in image convention
         shape is (nrows, ncols, 2) if mode==centers
         shape is (nrows+1, ncols+1, 2) if mode==edges

    """
    if mode == "centers":
        x = np.linspace(0.5, nx-0.5, num=nx)
        y = np.linspace(0.5, ny-0.5, num=ny)

    if mode == "edges":
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
    coordinates_map[..., 0] = (geotransform[0] +
                               geotransform[1] * coordinates[..., 0] +
                               geotransform[2] * coordinates[..., 1])
    coordinates_map[..., 1] = (geotransform[3] +
                               geotransform[4] * coordinates[..., 0] +
                               geotransform[5] * coordinates[..., 1])
    return coordinates_map


def read_gdal_coordinates(dataset, mode="centers"):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing
    mode : string
        either 'centers' or 'edges'

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
        Array of shape (nrows,ncols,2) containing xy coordinates.
        The array indexing follows image convention with origin
        at upper left pixel.
        The shape is (nrows+1,ncols+1,2) if mode == edges.

    Examples
    --------

    See :ref:`/notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    coordinates_pixel = _pixel_coordinates(dataset.RasterXSize,
                                           dataset.RasterYSize,
                                           mode)

    geotransform = dataset.GetGeoTransform()
    coordinates = _pixel_to_map(coordinates_pixel, geotransform)

    return coordinates


def read_gdal_projection(dataset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing

    Returns
    -------
    srs : OSR.SpatialReference
        dataset projection object

    Examples
    --------

    See :ref:`/notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    wkt = dataset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    # src = None
    return srs


def read_gdal_values(dataset=None, nodata=None):
    """Read values from a gdal object.

    Parameters
    ----------
    dataset : gdal.Dataset
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

    See :ref:`/notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

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


def extract_raster_dataset(dataset, mode="centers", nodata=None):
    """ Extract data, coordinates and projection information

    Parameters
    ----------
    dataset : gdal.Dataset
        raster dataset
    mode : string
        either 'centers' or 'edges'
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
        The shape is (nrows+1,ncols+1,2) if mode == edges.
    projection : osr object
        Spatial reference system of the used coordinates.
    """

    values = read_gdal_values(dataset, nodata=nodata)

    coords = read_gdal_coordinates(dataset, mode=mode)

    projection = read_gdal_projection(dataset)

    return values, coords, projection


def get_raster_extent(dataset, geo=False, window=True):
    """Get the coordinates of the 4 corners of the raster dataset

    Parameters
    ----------
    dataset : gdal.Dataset
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

    extent = np.array([[xmin, ymax],
                       [xmin, ymin],
                       [xmax, ymin],
                       [xmax, ymax]])

    if geo:
        projection = read_gdal_projection(dataset)
        fun = georef.projection.reproject
        extent = fun(extent, projection_source=projection)

    if window:
        x = extent[:, 0]
        y = extent[:, 1]
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

    return(extent)


def get_raster_elevation(dataset, **kwargs):
    """Return surface elevation corresponding to raster dataset
       The resampling algorithm is chosen based on scale ratio

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)
    kwargs : keyword arguments
        passed to wradlib.io.dem.get_strm()

    Returns
    -------
    elevation : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing elevation
    """
    extent = get_raster_extent(dataset)
    src_ds = wradlib.io.dem.get_srtm(extent, **kwargs)

    driver = gdal.GetDriverByName('MEM')
    dst_ds = driver.CreateCopy('ds', dataset)

    src_gt = src_ds.GetGeoTransform()
    dst_gt = src_ds.GetGeoTransform()
    src_scale = min(abs(src_gt[1]), abs(src_gt[5]))
    dst_scale = min(abs(dst_gt[1]), abs(dst_gt[5]))
    ratio = dst_scale/src_scale
    resample = gdal.GRA_Bilinear
    if ratio > 2:
        resample = gdal.GRA_Average
    if ratio < 0.5:
        resample = gdal.GRA_NearestNeighbour

    gdal.ReprojectImage(src_ds, dst_ds,
                        src_ds.GetProjection(), dst_ds.GetProjection(),
                        resample)
    elevation = read_gdal_values(dst_ds)

    return(elevation)


def set_raster_origin(data, coords, direction):
    """ Converts Data and Coordinates Origin

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
    origin = ('lower' if y_sp > 0 else 'upper')
    same = (origin == direction)
    if not same:
        data = np.flip(data, axis=-2)
        coords = np.flip(coords, axis=-3)
        # we need to shift y-coordinate if data and coordinates have the same
        # number of rows and cols (only the ll or ul raster coords are given)
#        if data.shape[-2:] == coords.shape[:2]:
#            coords += [0, y_sp]

    return data, coords


def reproject_raster_dataset(src_ds, **kwargs):
    """Reproject/Resample given dataset according to keyword arguments

    Parameters
    ----------
    src_ds : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)

    Keyword Arguments
    -----------------
    spacing : float
        float or tuple of two floats
        pixel spacing of destination dataset, same unit as pixel coordinates
    size : int
        tuple of two ints
        X/YRasterSize of destination dataset
    resample : GDALResampleAlg
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12
    projection_target : osr object
        destination dataset projection, defaults to None
    align : bool or Point
        If False, there is no destination grid aligment.
        If True, aligns the destination grid to the next integer multiple of
        destination grid.
        If Point (tuple, list of upper-left x,y-coordinate), the destination
        grid is aligned to this point.

    Returns
    -------
    dst_ds : gdal.Dataset
        reprojected/resampled raster dataset
    """

    # checking kwargs
    spacing = kwargs.pop('spacing', None)
    size = kwargs.pop('size', None)
    resample = kwargs.pop('resample', gdal.GRA_Bilinear)
    src_srs = kwargs.pop('projection_source', None)
    dst_srs = kwargs.pop('projection_target', None)
    align = kwargs.pop('align', False)

    # Get the GeoTransform vector
    src_geo = src_ds.GetGeoTransform()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    # get extent
    ulx = src_geo[0]
    uly = src_geo[3]
    lrx = src_geo[0] + src_geo[1] * x_size
    lry = src_geo[3] + src_geo[5] * y_size

    extent = np.array([[[ulx, uly],
                        [lrx, uly]],
                       [[ulx, lry],
                        [lrx, lry]]])

    if dst_srs:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_ds.GetProjection())

        # Transformation

        fun = georef.projection.reproject
        extent = fun(extent, projection_source=src_srs,
                     projection_target=dst_srs)

        # wkt needed
        src_srs = src_srs.ExportToWkt()
        dst_srs = dst_srs.ExportToWkt()

    (ulx, uly, urx, ury,
     llx, lly, lrx, lry) = tuple(list(extent.flatten().tolist()))

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
    else:
        raise NameError("Whether keyword 'spacing' or 'size' must be given")

    # create destination in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    # and set RasterSize according ro cols/rows
    dst_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Float32)

    # Create the destination GeoTransform with changed x/y spacing
    dst_geo = (ulx, x_ps, src_geo[2], uly, src_geo[4], -y_ps)

    # apply GeoTransform to destination dataset
    dst_ds.SetGeoTransform(dst_geo)

    # apply Projection to destination dataset
    if dst_srs is not None:
        dst_ds.SetProjection(dst_srs)

    # nodata handling, need to initialize dst_ds with nodata
    src_band = src_ds.GetRasterBand(1)
    nodata = src_band.GetNoDataValue()
    dst_band = dst_ds.GetRasterBand(1)
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)
        dst_band.WriteArray(np.ones((rows, cols)) * nodata)
    dst_band.FlushCache()

    # resample and reproject dataset
    gdal.ReprojectImage(src_ds, dst_ds, src_srs, dst_srs, resample)

    return dst_ds


def create_raster_dataset(data, coords, projection=None, nodata=-9999):
    """ Create In-Memory Raster Dataset

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    projection : osr object
        Spatial reference system of the used coordinates, defaults to None.
    nodata : int
        Value of NODATA

    Returns
    -------
    dataset : gdal.Dataset
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
    mem_drv = gdal.GetDriverByName('MEM')
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
    dataset = mem_drv.Create('', cols, rows, bands, gdal_type)

    # initialize geotransform
    x_ps, y_ps = coords[1, 1] - coords[0, 0]
    upper_corner_x = coords[0, 0, 0] - x_ps/2
    upper_corner_y = coords[0, 0, 1] - y_ps/2
    geotran = [upper_corner_x, x_ps, 0, upper_corner_y, 0, y_ps]
    dataset.SetGeoTransform(geotran)

    if projection:
        dataset.SetProjection(projection.ExportToWkt())

    # set np.nan to nodata
    dataset.GetRasterBand(1).SetNoDataValue(nodata)

    for i, band in enumerate(data, start=1):
        dataset.GetRasterBand(i).WriteArray(band)

    return dataset


def merge_rasters(datasets):
    """Merge rasters.

    Parameters
    ----------
    datasets : list of gdal.Dataset
        raster images with georeferencing

    Returns
    -------
    dataset : gdal.Dataset
        merged raster dataset
    """

    dataset = gdal.Warp('', datasets, format='MEM')

    return(dataset)


def raster_to_polyvert(dataset):
    """Get raster polygonal vertices from gdal dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)

    Returns
    -------
    polyvert : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape (..., 5, 2).

    """
    rastercoords = read_gdal_coordinates(dataset, mode="edges")

    polyvert = util.grid_to_polyvert(rastercoords)

    return(polyvert)
