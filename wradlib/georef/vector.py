#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Vector Functions (GDAL)
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "get_vector_coordinates",
    "get_vector_points",
    "transform_geometry",
    "ogr_create_layer",
    "ogr_copy_layer",
    "ogr_copy_layer_by_name",
    "ogr_reproject_layer",
    "ogr_add_feature",
    "ogr_add_geometry",
    "numpy_to_ogr",
    "ogr_to_numpy",
    "ogr_geocol_to_numpy",
    "get_centroid",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

import numpy as np

from wradlib.georef import projection
from wradlib.util import has_import, import_optional

gdal = import_optional("osgeo.gdal")
ogr = import_optional("osgeo.ogr")
osr = import_optional("osgeo.osr")

if has_import(gdal):
    ogr.UseExceptions()
    gdal.UseExceptions()


def get_vector_points(geom):
    """Extract coordinate points from given ogr geometry as generator object

    If geometries are nested, function recurses.

    Parameters
    ----------
    geom : :py:class:`gdal:osgeo.ogr.Geometry`

    Yields
    ------
    result : :py:class:`numpy:numpy.ndarray`
        expands to Nx2 dimensional nested point arrays
    """
    geomtype = geom.GetGeometryType()
    if geomtype > 1:
        # 1D Geometries, LINESTRINGS
        if geomtype == 2:
            result = np.array(geom.GetPoints())
            yield result
        # RINGS, POLYGONS, MULTIPOLYGONS, MULTILINESTRINGS
        elif geomtype > 2:
            # iterate over geometries and recurse
            for item in geom:
                for result in get_vector_points(item):
                    yield result
    else:
        warnings.warn(
            "unsupported geometry type detected in "
            "wradlib.georef.get_vector_points - skipping"
        )


def transform_geometry(geom, dest_srs, **kwargs):
    """Perform geotransformation to given destination SpatialReferenceSystem

    It transforms coordinates to a given destination osr spatial reference
    if a geotransform is neccessary.

    Parameters
    ----------
    geom : :py:class:`gdal:osgeo.ogr.Geometry`
    dest_srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection

    Keyword Arguments
    -----------------
    source_srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Source Projection

    Returns
    -------
    geom : :py:class:`gdal:osgeo.ogr.Geometry`
        Transformed Geometry
    """
    gsrs = geom.GetSpatialReference()
    srs = kwargs.get("source_srs", gsrs)

    # srs is None assume wgs84 lonlat, but warn too
    if srs is None:
        srs = projection.get_default_projection()
        warnings.warn("geometry without spatial reference - assuming wgs84")

    # transform if not the same spatial reference system
    if not srs.IsSame(dest_srs):
        if gsrs is None:
            geom.AssignSpatialReference(srs)
            gsrs = geom.GetSpatialReference()
        dest_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gsrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geom.TransformTo(dest_srs)

    return geom


def get_vector_coordinates(layer, **kwargs):
    """Function iterates over gdal ogr layer features and packs extracted \
    vector coordinate points into nested ndarray

    It transforms coordinates to a given destination osr spatial reference if
    dest_srs is given and a geotransform is necessary.

    Parameters
    ----------
    layer : :py:class:`gdal:osgeo.ogr.Layer`

    Keyword Arguments
    -----------------
    source_srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Source Projection
    dest_srs: :py:class:`gdal:osgeo.osr.SpatialReference`
        Destination Projection
    key : str
        attribute key to extract from layer feature

    Returns
    -------
    shp : :class:`numpy:numpy.ndarray`
        Dimension of subarrays Nx2
        extracted shape coordinate points
    attrs : list
        List of attributes extracted from features
    """

    shp = []

    source_srs = kwargs.get("source_srs", layer.GetSpatialRef())
    if source_srs is None:
        raise ValueError(
            "Spatial reference missing from source layer. "
            "Please provide a fitting spatial reference object"
        )

    dest_srs = kwargs.get("dest_srs", None)
    key = kwargs.get("key", None)
    if key:
        attrs = []
    else:
        attrs = None

    for i in range(layer.GetFeatureCount()):
        feature = layer.GetNextFeature()
        if feature:
            if key:
                attrs.append(feature[key])
            geom = feature.GetGeometryRef()
            if dest_srs:
                transform_geometry(geom, dest_srs, source_srs=source_srs)
            # get list of xy-coordinates
            reslist = list(get_vector_points(geom))
            shp.append(np.squeeze(np.array(reslist)))

    shp = np.squeeze(np.array(shp, dtype=object))

    return shp, attrs


def ogr_reproject_layer(src_lyr, dst_lyr, dst_srs, src_srs=None):
    """Reproject src_lyr to dst_lyr.

    Creates one OGR.Layer with given name in given gdal.Dataset object
    using given OGR.GeometryType and FieldDefinitions

    Parameters
    ----------
    src_lyr : :py:class:`gdal:osgeo.ogr.Layer`
        OGRLayer source layer
    dst_lyr : :py:class:`gdal:osgeo.ogr.Layer`
        OGRLayer destination layer
    dst_srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Projection Target SRS
    src_srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        Projection Source SRS

    Returns
    -------
    dst_lyr : :py:class:`gdal:osgeo.ogr.Layer`
        OGRLayer destination layer
    """
    if src_srs is None:
        src_srs = src_lyr.GetSpatialRef()
        if src_srs is None:
            raise ValueError(
                "Spatial reference missing from source layer. "
                "Please provide a fitting spatial reference object"
            )

    # add fields
    dst_lyr.CreateField(ogr.FieldDefn("index", ogr.OFTInteger))

    # get the output layer's feature definition
    dst_lyr_defn = dst_lyr.GetLayerDefn()
    # loop through the input features
    src_feature = src_lyr.GetNextFeature()
    i = 0
    while src_feature:
        # get the input geometry
        geom = src_feature.GetGeometryRef()
        # reproject the geometry
        geom = transform_geometry(geom, source_srs=src_srs, dest_srs=dst_srs)
        # create a new feature
        dst_feature = ogr.Feature(dst_lyr_defn)
        # set the geometry and attribute
        dst_feature.SetGeometry(geom)
        dst_feature.SetField("index", i)
        i += 1
        # add the feature to the shapefile
        dst_lyr.CreateFeature(dst_feature)
        # dereference the features and get the next input feature
        dst_feature = None
        src_feature = src_lyr.GetNextFeature()

    return dst_lyr


def ogr_create_layer(ds, name, srs=None, geom_type=None, fields=None):
    """Creates OGR.Layer objects in gdal.Dataset object.

    Creates one OGR.Layer with given name in given gdal.Dataset object
    using given OGR.GeometryType and FieldDefinitions

    Parameters
    ----------
    ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    name : str
        OGRLayer name
    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        object
    geom_type : :py:class:`gdal:osgeo.ogr.GeometryType`
        (eg. ogr.wkbPolygon)
    fields : list
        list of 2 element tuples
        (str, :py:class:`gdal:osgeo.ogr.DataType`) field name, field type

    Returns
    -------
    out : :py:class:`gdal:osgeo.ogr.Layer`
        object
    """
    if geom_type is None:
        raise TypeError("geometry_type needed")

    lyr = ds.CreateLayer(name, srs=srs, geom_type=geom_type)
    if fields is not None:
        for fname, fvalue in fields:
            lyr.CreateField(ogr.FieldDefn(fname, fvalue))

    return lyr


def ogr_copy_layer(src_ds, index, dst_ds, reset=True):
    """Copy OGR.Layer object.

    Copy OGR.Layer object from src_ds gdal.Dataset to dst_ds gdal.Dataset

    Parameters
    ----------
    src_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    index : int
        layer index
    dst_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    reset : bool
        if True resets src_layer
    """
    # get and copy src geometry layer

    src_lyr = src_ds.GetLayerByIndex(index)
    if reset:
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        src_lyr.SetAttributeFilter(None)
    dst_ds.CopyLayer(src_lyr, src_lyr.GetName())


def ogr_copy_layer_by_name(src_ds, name, dst_ds, reset=True):
    """Copy OGR.Layer object.

    Copy OGR.Layer object from src_ds gdal.Dataset to dst_ds gdal.Dataset

    Parameters
    ----------
    src_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    name : str
        layer name
    dst_ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    reset : bool
        if True resets src_layer
    """
    # get and copy src geometry layer

    src_lyr = src_ds.GetLayerByName(name)
    if src_lyr is None:
        raise ValueError("OGR layer 'name' not found in dataset")
    if reset:
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        src_lyr.SetAttributeFilter(None)
    dst_ds.CopyLayer(src_lyr, src_lyr.GetName())


def ogr_add_feature(ds, src, name=None):
    """Creates OGR.Feature objects in OGR.Layer object.

    OGR.Features are built from numpy src points or polygons.

    OGR.Features 'FID' and 'index' corresponds to source data element

    Parameters
    ----------
    ds : :py:class:`gdal:osgeo.gdal.Dataset`
        object
    src : :py:class:`numpy:numpy.ndarray`
        source data
    name : str
        name of wanted Layer
    """

    if name is not None:
        lyr = ds.GetLayerByName(name)
    else:
        lyr = ds.GetLayer()

    defn = lyr.GetLayerDefn()
    geom_name = ogr.GeometryTypeToName(lyr.GetGeomType())
    fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    feat = ogr.Feature(defn)

    for index, src_item in enumerate(src):
        geom = numpy_to_ogr(src_item, geom_name)

        if "index" in fields:
            feat.SetField("index", index)

        feat.SetGeometry(geom)
        lyr.CreateFeature(feat)


def ogr_add_geometry(layer, geom, attrs):
    """Copies single OGR.Geometry object to an OGR.Layer object.

    Given OGR.Geometry is copied to new OGR.Feature and
    written to given OGR.Layer by given index. Attributes are attached.

    Parameters
    ----------
    layer : :py:class:`gdal:osgeo.ogr.Layer`
        object
    geom : :py:class:`gdal:osgeo.ogr.Geometry`
        object
    attrs : list
        attributes referring to layer fields

    """
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)

    for i, item in enumerate(attrs):
        feat.SetField(i, item)
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)


def numpy_to_ogr(vert, geom_name):
    """Convert a vertex array to gdal/ogr geometry.

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    vert : array-like
        a numpy array of vertices of shape (num vertices, 2)
    geom_name : str
        Name of Geometry

    Returns
    -------
    out : :py:class:`gdal:osgeo.ogr.Geometry`
        object of type geom_name
    """

    if geom_name in ["Polygon", "MultiPolygon"]:
        json_str = f"{{'type':'{geom_name}','coordinates':[{repr(vert.tolist())}]}}"
    else:
        json_str = f"{{'type':'{geom_name}','coordinates':{repr(vert.tolist())}}}"
    return ogr.CreateGeometryFromJson(json_str)


def ogr_to_numpy(ogrobj):
    """Backconvert a gdal/ogr geometry to a numpy vertex array.

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    ogrobj : :py:class:`gdal:osgeo.ogr.Geometry`
        object

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        a nested ndarray of vertices of shape (num vertices, 2)

    """
    jsonobj = eval(ogrobj.ExportToJson())

    return np.squeeze(jsonobj["coordinates"])


def ogr_geocol_to_numpy(ogrobj):
    """Backconvert a gdal/ogr geometry Collection to a numpy vertex array.

    This extracts only Polygon geometries!

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    ogrobj : :py:class:`gdal:osgeo.ogr.Geometry`
        Collection object

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        a nested ndarray of vertices of shape (num vertices, 2)

    """
    jsonobj = eval(ogrobj.ExportToJson())
    mpol = []
    for item in jsonobj["geometries"]:
        if item["type"] == "Polygon":
            mpol.append(item["coordinates"])

    return np.squeeze(mpol)


def get_centroid(polyg):
    """Return centroid of a polygon

    Parameters
    ----------
    polyg : :class:`numpy:numpy.ndarray`
        of shape (num vertices, 2) or ogr.Geometry object

    Returns
    -------
    out : tuple
        x and y coordinate of the centroid

    """
    if not type(polyg) == ogr.Geometry:
        polyg = numpy_to_ogr(polyg, "Polygon")
    return polyg.Centroid().GetPoint()[0:2]
