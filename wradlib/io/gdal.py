#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raster and Vector I/O using GDAL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "open_vector",
    "open_raster",
    "read_safnwc",
    "gdal_create_dataset",
    "write_raster_dataset",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import os

from osgeo import gdal, osr


def open_vector(filename, driver=None, layer=0):
    """Open vector file, return gdal.Dataset and OGR.Layer

        .. warning:: dataset and layer have to live in the same context,
            if dataset is deleted all layer references will get lost

    Parameters
    ----------
    filename : string
        vector file name
    driver : string
        gdal driver string
    layer : int or string

    Returns
    -------
    dataset : gdal.Dataset
        dataset
    layer : ogr.Layer
        layer
    """
    dataset = gdal.OpenEx(filename)

    if driver:
        gdal.GetDriverByName(driver)

    layer = dataset.GetLayer(layer)

    return dataset, layer


def open_raster(filename, driver=None):
    """Open raster file, return gdal.Dataset

    Parameters
    ----------
    filename : string
        raster file name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : gdal.Dataset
        dataset
    """

    dataset = gdal.OpenEx(filename)

    if driver:
        gdal.GetDriverByName(driver)

    return dataset


def read_safnwc(filename):
    """Read MSG SAFNWC hdf5 file into a gdal georeferenced object

    Parameters
    ----------
    filename : string
        satellite file name

    Returns
    -------
    ds : gdal.DataSet
        with satellite data
    """

    root = gdal.Open(filename)
    ds1 = gdal.Open("HDF5:" + filename + "://CT")
    ds = gdal.GetDriverByName("MEM").CreateCopy("out", ds1, 0)

    try:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(ds.GetMetadata()["PROJECTION"])
    except KeyError:
        raise KeyError(
            "WRADLIB: Projection is missing for satellite " "file {}".format(filename)
        )

    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj.ExportToWkt())
    ds.SetGeoTransform([float(x) for x in geotransform])

    return ds


def gdal_create_dataset(
    drv, name, cols=0, rows=0, bands=0, gdal_type=gdal.GDT_Unknown, remove=False
):
    """Creates GDAL.DataSet object.

    Parameters
    ----------
    drv : string
        GDAL driver string
    name : string
        path to filename
    cols : int
        # of columns
    rows : int
        # of rows
    bands : int
        # of raster bands
    gdal_type : raster data type
        eg. gdal.GDT_Float32
    remove : bool
        if True, existing gdal.Dataset will be
        removed before creation

    Returns
    -------
    out : gdal.Dataset
        object

    """
    driver = gdal.GetDriverByName(drv)
    metadata = driver.GetMetadata()

    if not metadata.get("DCAP_CREATE", False):
        raise TypeError(
            "WRADLIB: Driver {} doesn't support " "Create() method.".format(drv)
        )

    if remove:
        if os.path.exists(name):
            driver.Delete(name)
    ds = driver.Create(name, cols, rows, bands, gdal_type)

    return ds


def write_raster_dataset(fpath, dataset, rformat, options=None, remove=False):
    """Write raster dataset to file format

    Parameters
    ----------
    fpath : string
        A file path - should have file extension corresponding to format.
    dataset : gdal.Dataset
        gdal raster dataset
    rformat : string
        gdal raster format string
    options : list
        List of option strings for the corresponding format.
    remove : bool
        if True, existing gdal.Dataset will be
        removed before creation

    Note
    ----
    For format and options refer to
    `formats_list <http://www.gdal.org/formats_list.html>`_.

    Examples
    --------
    See :ref:`/notebooks/fileio/wradlib_gis_export_example.ipynb`.
    """
    # check for option list
    if options is None:
        options = []

    driver = gdal.GetDriverByName(rformat)
    metadata = driver.GetMetadata()

    # check driver capability
    if not ("DCAP_CREATECOPY" in metadata and metadata["DCAP_CREATECOPY"] == "YES"):
        raise TypeError(
            "WRADLIB: Raster Driver {} doesn't support "
            "CreateCopy() method.".format(rformat)
        )

    if remove:
        if os.path.exists(fpath):
            driver.Delete(fpath)

    target = driver.CreateCopy(fpath, dataset, 0, options)
    del target
