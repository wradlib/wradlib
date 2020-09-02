#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
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
from osgeo import gdal, ogr, osr

from wradlib.georef import projection

ogr.UseExceptions()
gdal.UseExceptions()


def get_vector_points(geom):
    """Extract coordinate points from given ogr geometry as generator object

    If geometries are nested, function recurses.

    Parameters
    ----------
    geom : ogr.Geometry

    Returns
    -------
    result : generator object
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
    geom : ogr.geometry
    dest_srs : osr.SpatialReference
        Destination Projection

    Keyword Arguments
    -----------------
    source_srs : osr.SpatialReference
        Source Projection

    Returns
    -------
    geom : ogr.Geometry
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
        if gdal.VersionInfo()[0] >= "3":
            dest_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            gsrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geom.TransformTo(dest_srs)

    return geom


def get_vector_coordinates(layer, **kwargs):
    """Function iterates over gdal ogr layer features and packs extracted \
    vector coordinate points into nested ndarray

    It transforms coordinates to a given destination osr spatial reference if
    dest_srs is given and a geotransform is neccessary.

    Parameters
    ----------
    layer : ogr.Layer

    Keyword Arguments
    -----------------
    source_srs : osr.SpatialReference
        Source Projection
    dest_srs: osr.SpatialReference
        Destination Projection
    key : string
        attribute key to extract from layer feature

    Returns
    -------
    shp : nested :class:`numpy:numpy.ndarray`
        Dimension of subarrays Nx2
        extracted shape coordinate points
    attrs : list
        List of attributes extracted from features
    """

    shp = []

    source_srs = kwargs.get("source_srs", None)
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


def ogr_create_layer(ds, name, srs=None, geom_type=None, fields=None):
    """Creates OGR.Layer objects in gdal.Dataset object.

    Creates one OGR.Layer with given name in given gdal.Dataset object
    using given OGR.GeometryType and FieldDefinitions

    Parameters
    ----------
    ds : gdal.Dataset
        object
    name : string
        OGRLayer name
    srs : OSR.SpatialReference
        object
    geom_type : OGR GeometryType
        (eg. ogr.wkbPolygon)
    fields : list of 2 element tuples
        (strings, OGR.DataType) field name, field type

    Returns
    -------
    out : OGR.Layer
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
    src_ds : gdal.Dataset
        object
    index : int
        layer index
    dst_ds : gdal.Dataset
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
    src_ds : gdal.Dataset
        object
    name : string
        layer name
    dst_ds : gdal.Dataset
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
    ds : gdal.Dataset
        object
    src : :func:`numpy:numpy.array`
        source data
    name : string
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
    layer : OGR.Layer
        object
    geom : OGR.Geometry
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
    vert : array_like
        a numpy array of vertices of shape (num vertices, 2)
    geom_name : string
        Name of Geometry

    Returns
    -------
    out : ogr.Geometry
        object of type geom_name
    """

    if geom_name in ["Polygon", "MultiPolygon"]:
        json_str = "{{'type':{0!r},'coordinates':[{1!r}]}}".format(
            geom_name, vert.tolist()
        )
    else:
        json_str = "{{'type':{0!r},'coordinates':{1!r}}}".format(
            geom_name, vert.tolist()
        )

    return ogr.CreateGeometryFromJson(json_str)


def ogr_to_numpy(ogrobj):
    """Backconvert a gdal/ogr geometry to a numpy vertex array.

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    ogrobj : ogr.Geometry
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
    ogrobj : ogr.Geometry
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
    out : x and y coordinate of the centroid

    """
    if not type(polyg) == ogr.Geometry:
        polyg = numpy_to_ogr(polyg, "Polygon")
    return polyg.Centroid().GetPoint()[0:2]
