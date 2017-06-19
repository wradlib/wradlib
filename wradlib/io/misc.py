#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous Data I/O
^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   writePolygon2Text
   read_safnwc
   write_raster_dataset
   to_pickle
   from_pickle
   open_shape
   open_raster
"""

# standard libraries
from __future__ import absolute_import

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os

# site packages
from osgeo import gdal, ogr, osr


def _write_polygon2txt(f, idx, vertices):
    f.write('%i %i\n' % idx)
    for i, vert in enumerate(vertices):
        f.write('%i ' % (i,))
        f.write('%f %f %f %f\n' % tuple(vert))


def writePolygon2Text(fname, polygons):
    """Writes Polygons to a Text file which can be interpreted by ESRI \
    ArcGIS's "Create Features from Text File (Samples)" tool.

    This is (yet) only a convenience function with limited functionality.
    E.g. interior rings are not yet supported.

    Parameters
    ----------
    fname : string
        name of the file to save the vertex data to
    polygons : list of lists
        list of polygon vertices.
        Each vertex itself is a list of 3 coordinate values and an
        additional value. The third coordinate and the fourth value may be nan.

    Returns
    -------
    None

    Note
    ----
    As Polygons are closed shapes, the first and the last vertex of each
    polygon **must** be the same!

    Examples
    --------
    Writes two triangle Polygons to a text file::

        poly1 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        poly2 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        polygons = [poly1, poly2]
        writePolygon2Text('polygons.txt', polygons)

    The resulting text file will look like this::

        Polygon
        0 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        1 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        END

    """
    with open(fname, 'w') as f:
        f.write('Polygon\n')
        count = 0
        for vertices in polygons:
            _write_polygon2txt(f, (count, 0), vertices)
            count += 1
        f.write('END\n')


def to_pickle(fpath, obj):
    """Pickle object <obj> to file <fpath>
    """
    output = open(fpath, 'wb')
    pickle.dump(obj, output)
    output.close()


def from_pickle(fpath):
    """Return pickled object from file <fpath>
    """
    pkl_file = open(fpath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    return obj


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
    ds1 = gdal.Open('HDF5:' + filename + '://CT')
    ds = gdal.GetDriverByName('MEM').CreateCopy('out', ds1, 0)

    # name = os.path.basename(filename)[7:11]
    try:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(ds.GetMetadata()["PROJECTION"])
    except Exception:
        raise NameError("No metadata for satellite file %s" % filename)
    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj.ExportToWkt())
    ds.SetGeoTransform([float(x) for x in geotransform])
    return ds


def write_raster_dataset(fpath, dataset, format, options=None, remove=False):
    """ Write raster dataset to file format

    .. versionadded 0.10.0

    Parameters
    ----------
    fpath : string
        A file path - should have file extension corresponding to format.
    dataset : gdal.Dataset
        gdal raster dataset
    format : string
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
    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.
    """
    # check for option list
    if options is None:
        options = []

    driver = gdal.GetDriverByName(format)
    metadata = driver.GetMetadata()

    # check driver capability
    if 'DCAP_CREATECOPY' in metadata and metadata['DCAP_CREATECOPY'] != 'YES':
        assert "Driver %s doesn't support CreateCopy() method.".format(format)

    if remove:
        if os.path.exists(fpath):
            driver.Delete(fpath)

    target = driver.CreateCopy(fpath, dataset, 0, options)
    del target


def open_shape(filename, driver=None):
    """
    Open shapefile, return gdal.Dataset and OGR.Layer

    .. warning:: dataset and layer have to live in the same context,
                 if dataset is deleted all layer references will get lost

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        shapefile name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : gdal.Dataset
        dataset
    layer : ogr.Layer
        layer
    """

    if driver is None:
        driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(filename)
    if dataset is None:
        print('Could not open file')
        raise IOError
    layer = dataset.GetLayer()
    return dataset, layer


def open_raster(filename, driver=None):
    """
    Open raster file, return gdal.Dataset

    .. versionadded:: 0.6.0

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

    dataset = gdal.Open(filename)

    if driver:
        gdal.GetDriverByName(driver)

    return dataset


if __name__ == '__main__':
    print('wradlib: Calling module <io> as main...')
