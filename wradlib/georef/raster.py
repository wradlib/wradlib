#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2017, wradlib developers.
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
   pixel_to_map3d
   pixel_to_map
   pixel_coordinates
   reproject_raster_dataset
   create_raster_dataset
   set_raster_origin
   extract_raster_dataset
"""

import numpy as np
from osgeo import gdal, osr, gdal_array

from .projection import reproject


def pixel_coordinates(nx, ny, mode="centers"):
    """Get pixel coordinates from a regular grid with dimension nx by ny.

    Parameters
    ----------
    nx : int
        xsize
    ny : int
        ysize
    mode : string
        `centers` or `centroids` to return the pixel centers coordinates
        otherwise the pixel edges coordinates will be returned
    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
         Array of shape (ny,nx) with pixel coordinates (x,y)

    """
    if mode == "centroids":
        mode = "centers"
    x = np.linspace(0, nx, num=nx + 1)
    y = np.linspace(0, ny, num=ny + 1)
    if mode == "centers":
        x = x + 0.5
        y = y + 0.5
        x = np.delete(x, -1)
        y = np.delete(y, -1)
    X, Y = np.meshgrid(x, y)
    coordinates = np.empty(X.shape + (2,))
    coordinates[:, :, 0] = X
    coordinates[:, :, 1] = Y
    return (coordinates)


def pixel_to_map(geotransform, coordinates):
    """Apply a geographical transformation to return map coordinates from
    pixel coordinates.

    Parameters
    ----------
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector:

            - geotransform[0] = East/West location of Upper Left corner
            - geotransform[1] = X pixel size
            - geotransform[2] = X pixel rotation
            - geotransform[3] = North/South location of Upper Left corner
            - geotransform[4] = Y pixel rotation
            - geotransform[5] = Y pixel size
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        3d array with map coordinates x,y
    """
    coordinates_map = np.empty(coordinates.shape)
    coordinates_map[..., 0] = (geotransform[0] +
                               geotransform[1] * coordinates[..., 0] +
                               geotransform[2] * coordinates[..., 1])
    coordinates_map[..., 1] = (geotransform[3] +
                               geotransform[4] * coordinates[..., 0] +
                               geotransform[5] * coordinates[..., 1])
    return (coordinates_map)


def pixel_to_map3d(geotransform, coordinates, z=None):
    """Apply a geographical transformation to return 3D map coordinates from
    pixel coordinates.

    Parameters
    ----------
    geotransform : :class:`numpy:numpy.ndarray`
        geographical transformation vector
        (see :meth:`~wradlib.georef.pixel_to_map`)
    coordinates : :class:`numpy:numpy.ndarray`
        2d array of pixel coordinates;
    z : string
        method to compute the z coordinates (height above ellipsoid):

            - None : default, z equals zero
            - srtm : not available yet

    Returns
    -------
    coordinates_map : :class:`numpy:numpy.ndarray`
        4d array with map coordinates x,y,z

    """

    coordinates_map = np.empty(coordinates.shape[:-1] + (3,))
    coordinates_map[..., 0:2] = pixel_to_map(geotransform, coordinates)
    coordinates_map[..., 2] = np.zeros(coordinates.shape[:-1])
    return (coordinates_map)


def read_gdal_coordinates(dataset, mode='centers', z=True):
    """Get the projected coordinates from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        raster image with georeferencing
    mode : string
        either 'centers' or 'borders'
    z : boolean
        True to get height coordinates (zero).

    Returns
    -------
    coordinates : :class:`numpy:numpy.ndarray`
        Array of projected coordinates (x,y,z)

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    coordinates_pixel = pixel_coordinates(dataset.RasterXSize,
                                          dataset.RasterYSize, mode)
    geotransform = dataset.GetGeoTransform()
    if z:
        coordinates = pixel_to_map3d(geotransform, coordinates_pixel)
    else:
        coordinates = pixel_to_map(geotransform, coordinates_pixel)
    return (coordinates)


def read_gdal_projection(dset):
    """Get a projection (OSR object) from a GDAL dataset.

    Parameters
    ----------
    dset : gdal.Dataset

    Returns
    -------
    srs : OSR.SpatialReference
        dataset projection object

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

    """
    wkt = dset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    # src = None
    return srs


def read_gdal_values(dataset=None, nodata=None):
    """Read values from a gdal object.

    Parameters
    ----------
    data : gdal object
    nodata : boolean
        option to deal with nodata values replacing it with nans.

    Returns
    -------
    values : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.

    Examples
    --------

    See :ref:`notebooks/classify/wradlib_clutter_cloud_example.ipynb`.

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

    return np.squeeze(np.stack(bands))


def create_raster_dataset(data, coords, projection=None, nodata=-9999):
    """ Create In-Memory Raster Dataset

    .. versionadded 0.10.0

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    projection : osr object
        Spatial reference system of the used coordinates, defaults to None.

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
    geotran = [coords[0, 0, 0], x_ps, 0, coords[0, 0, 1], 0, y_ps]
    dataset.SetGeoTransform(geotran)

    if projection:
        dataset.SetProjection(projection.ExportToWkt())

    # set np.nan to nodata
    dataset.GetRasterBand(1).SetNoDataValue(nodata)

    for i, band in enumerate(data, start=1):
        dataset.GetRasterBand(i).WriteArray(band)

    return dataset


def set_raster_origin(data, coords, direction):
    """ Converts Data and Coordinates Origin

    .. versionadded 0.10.0

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
        # number of rows and cols
        if data.shape[-2:] == coords.shape[:2]:
            coords += [0, y_sp]

    return data, coords


def extract_raster_dataset(dataset, nodata=None):
    """ Extract data, coordinates and projection information

    Parameters
    ----------
    dataset : gdal.Dataset
        raster dataset
    nodata : scalar
        Value to which the dataset nodata values are mapped.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols) or (bands, rows, cols) containing
        the data values.
    coords : :class:`numpy:numpy.ndarray`
        Array of shape (rows, cols, 2) containing xy-coordinates.
    projection : osr object
        Spatial reference system of the used coordinates.
    """

    # data values
    data = read_gdal_values(dataset, nodata=nodata)

    # coords
    coords_pixel = pixel_coordinates(dataset.RasterXSize,
                                     dataset.RasterYSize,
                                     'edges')
    coords = pixel_to_map(dataset.GetGeoTransform(),
                          coords_pixel)

    projection = read_gdal_projection(dataset)

    return data, coords, projection


def reproject_raster_dataset(src_ds, **kwargs):
    """Reproject/Resample given dataset according to keyword arguments

    .. versionadded:: 0.10.0

    # function inspired from github project
    # https://github.com/profLewis/geogg122

    Parameters
    ----------
    src_ds : gdal.Dataset
        raster image with georeferencing (GeoTransform at least)
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
        print("dest_src available")
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_ds.GetProjection())

        # Transformation
        extent = reproject(extent, projection_source=src_srs,
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

    # nodata handling, need to initialize dst_ds with nodata
    src_band = src_ds.GetRasterBand(1)
    nodata = src_band.GetNoDataValue()
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(nodata)
    dst_band.WriteArray(np.ones((rows, cols)) * nodata)
    dst_band.FlushCache()

    # resample and reproject dataset
    gdal.ReprojectImage(src_ds, dst_ds, src_srs, dst_srs, resample)

    return dst_ds
