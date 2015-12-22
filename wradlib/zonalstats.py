#-------------------------------------------------------------------------------
# Name:        zonalstats
# Purpose:
#
# Author:      Maik Heistermann, Kai Muehlbauer
#
# Created:     12.11.2015
# Copyright:   (c) Maik Heistermann, Kai Muehlbauer 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Zonal Statistics
^^^^^^^^^^^^^^^^

.. versionadded:: 0.7.0

This module supports you in computing statistics over spatial zones. A typical
application would be to compute mean areal precipitation for a catchment by using
precipitation estimates from a radar grid in polar coordinates, or from precipitation
estimates in a Cartesian grid.

The general usage is similar to the ipol and adjustment modules: You have to
create an instance of a class by using the spatial information of your source and
target objects (e.g. radar bins and catchment polygons). You can then compute
zonal statistics for your target objects by calling the instance with an array of
values (one for each source object). Typically, creating the instance will be
computationally expensive, but only has to be done once (as long as the geometries
do not change). Calling the objects with actual data, however, will be very fast.

..note:: Right now we only support a limited set of 2-dimensional zonal statistics.
         In the future, we plan to extend this to three dimensions.


.. currentmodule:: wradlib.zonalstats

.. autosummary::
   :nosignatures:
   :toctree: generated/

   GridCellsToPoly
   GridPointsToPoly

"""

import os

from osgeo import ogr
import numpy as np
from scipy.spatial import cKDTree
import datetime as dt

import wradlib.io as io


class ZonalStatsBase():
    """Base class for all 2-dimensional zonal statistics.

    .. versionadded:: 0.7.0

    The base class for computing 2-dimensional zonal statistics for target
    polygons from source points or polygons. Provides the basic design
    for all other classes.
    
    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the zonal statistic for that target will be NaN.

    Parameters
    ----------
    src : src : sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons
    """
    # TODO: incorporate OGR.SpatialReference to get src and trg srs in line
    def __init__(self, src, trg=None, ix=None, w=None, buffer=0., **kwargs):

        self._ix = []
        self._w = []
        self._buffer = buffer

        # if only src is given assume "dump_all_shape" filename
        if trg is None and ix is None and w is None:
            self.load_all_shape(src)

            # TODO: put this in extra function for reading index and weight from dst_N-layer
            # get target features
            cnt = self.trg.GetLayerByName('trg_poly').GetFeatureCount()

            for i in range(cnt):
                lyr = self._dst.GetLayerByName('dst_{0}'.format(i))
                lyr.ResetReading()
                ix = [ogr_src.GetField('index') for ogr_src in lyr]
                lyr.ResetReading()
                w = [ogr_src.GetField('weight') for ogr_src in lyr]
                self._add_idx_weights(ix, w)

        else:
            self.src = self._check_src(src, **kwargs)

            if trg is not None:
                self._trg, self._dst = self._check_trg(trg, **kwargs)
            else:
                self._add_idx_weights(ix, w, **kwargs)


    # TODO: temporarily disabled
    # make this work for adding target polygons
    # is this really necessary?
    #def _add_target(self, trg, **kwargs):
    #    #print(trg)
    #
    #    ix, w = self.get_weights(trg, **kwargs)
    #    self._add_idx_weights(ix, w, **kwargs)

    def _add_idx_weights(self, ix, w, **kwargs):
        ix, w = self._check_ix_w(ix, w, **kwargs)
        self.ix = self.ix + [ix]
        self.w = self.w + [w]

    # TODO: check which properties are really needed
    @property
    def ix(self):
        return self._ix

    @ix.setter
    def ix(self, value):
        self._ix = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, value):
        self._buffer = value
        # TODO: recalculate index and weights

    @property
    def trg(self):
        return self._trg

    #@trg.setter
    #def trg(self, value):
    #    self._trg = value

    def dump_src_shape(self, filename, remove=True):
        """ Output source grid points/polygons to ESRI_Shapefile

        Parameters
        ----------
        filename : string, path to shape-filename

        """
        ds_out = ogr_create_datasource('ESRI Shapefile', filename, remove=remove)
        ogr_copy_layer(self.src, 'src_grid', ds_out)

        # flush everything
        del ds_out

    def dump_all_shape(self, filename, remove=True):
        """ Output source/target grid points/polygons to ESRI_Shapefile

        target layer features are attributed with source index and weight

        Parameters
        ----------
        filename : string, path to shape-filename

        """
        # create output file datasource
        ds_out = ogr_create_datasource('ESRI Shapefile', filename, remove=remove)

        # get and copy src geometry layer
        ogr_copy_layer(self.src, 'src_grid', ds_out)

        # get and copy target polygon layer
        ogr_copy_layer(self.trg, 'trg_poly', ds_out)

        # get and copy memory destination trg layers to output datasource
        [ogr_copy_layer(self._dst, 'dst_{0}'.format(i), ds_out) for i in range(self._dst.GetLayerCount())]

        # flush everything
        del ds_out

    def load_all_shape(self, filename):
        """ Load source/target grid points/polygons into in memory Shapefile

        Parameters
        ----------
        filename : string, path to shape-filename

        """
        # get input file handles
        ds_in, tmp = io.open_shape(filename)

        # claim memory driver
        drv_in = ogr.GetDriverByName('Memory')

        # create all DataSources
        self.src = drv_in.CreateDataSource('src')
        self._trg = drv_in.CreateDataSource('trg')
        self._dst = drv_in.CreateDataSource('dst')

        # get src and target polygon layer
        ogr_copy_layer(ds_in, 'src_grid', self.src)
        ogr_copy_layer(ds_in, 'trg_poly', self._trg)

        # get destination trg layers
        [ogr_copy_layer(ds_in, 'dst_{0}'.format(i), self._dst) for i in range(ds_in.GetLayerCount() - 2)]

        # flush everything
        del ds_in

    def get_weights(self, trg, **kwargs):
        """This is the key method that needs to be filled for any inheriting class.
        """
        pass

    def check_empty(self):
        """
        """
        isempty = np.repeat(False, len(self.w))
        for i, weights in enumerate(self.w):
            if np.sum(weights)==0 or np.isnan(np.sum(weights)):
                isempty[i] = True
        return isempty
        
    def _check_src(self, src, **kwargs):
        """ Basic check of source elements (sequence of points or polygons).

            - array cast of source elements
            - create ogr_src datasource/layer holding src points/polygons
            - transforming source grid points/polygons to ogr.geometries on ogr.layer

        """
        t1 = dt.datetime.now()

        ogr_src = ogr_create_datasource('Memory', 'src')

        try:
            # is it ESRI Shapefile?
            ds_in, tmp_lyr = io.open_shape(src, driver=ogr.GetDriverByName('ESRI Shapefile'))
            ogr_trg_lyr = ogr_src.CopyLayer(tmp_lyr, 'src_grid')
        except IOError:
            # no ESRI shape file
            raise
        # all failed? then it should be sequence or numpy array
        except RuntimeError:
            src = np.array(src)
            # create memory datasource, layer and create features
            if src.ndim == 3:
                geom_type = ogr.wkbPolygon
            # no Polygons, just Points
            else:
                geom_type = ogr.wkbPoint

            fields = {'index': ogr.OFTInteger}
            ogr_create_layer(ogr_src, 'src_grid', geom_type, fields)
            ogr_add(ogr_src, src, name='src_grid')

        t2 = dt.datetime.now()
        print "Setting up OGR Layer takes: %f seconds" % (t2 - t1).total_seconds()

        return ogr_src

    def _check_trg(self, trg, **kwargs):
        """ Basic check of target elements (sequence of points or polygons).

            Iterates over target elements (and transforms to ogr.Polygon if necessary)
            create ogr_trg datasource/layer holding target polygons

        """
        t1 = dt.datetime.now()
        # if no targets are given

        # create target polygon ogr.DataSource with dedicated target polygon layer
        ogr_trg = ogr_create_datasource('Memory', 'trg')

        try:
            # is it ESRI Shapefile?
            ds, tmp_lyr = io.open_shape(trg, driver=ogr.GetDriverByName('ESRI Shapefile'))
            ogr_trg_lyr = ogr_trg.CopyLayer(tmp_lyr, 'trg_poly')
        except IOError:
            # no ESRI shape file
            raise
        except RuntimeError:
            trg = np.array(trg)
            # create layer and features
            fields = {'index': ogr.OFTInteger}
            ogr_create_layer(ogr_trg, 'trg_poly', ogr.wkbPolygon, fields)
            ogr_add(ogr_trg, trg, name='trg_poly')

        # the following code creates the destination target datasource with one layer
        # for each target polygon, consisting of the needed source data attributed with
        # index and weights

        # create intermediate mem datasource
        ds_mem = ogr_create_datasource('Memory', 'dst')

        # get src geometry layer
        src_lyr = self.src.GetLayerByName('src_grid')
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        geom_type = src_lyr.GetGeomType()

        fields = {'index': ogr.OFTInteger, 'weight': ogr.OFTReal}

        # get target layer, iterate over polygons and calculate weights
        lyr = ogr_trg.GetLayer()
        lyr.ResetReading()
        for index, trg_poly in enumerate(lyr):

            # create layer as self.tmp_lyr
            self.tmp_lyr = ogr_create_layer(ds_mem, 'dst_{0}'.format(index), geom_type, fields)

            # calculate weights while filling self.tmp_lyr with index and weights information
            ix, w = self.get_weights(trg_poly.GetGeometryRef(), **kwargs)
            self._add_idx_weights(ix, w, **kwargs)

        t2 = dt.datetime.now()
        print "Setting up Target takes: %f seconds" % (t2 - t1).total_seconds()

        return ogr_trg, ds_mem

    def _check_ix_w(self, ix, w, **kwargs):
        """TODO Basic check of target attributes (sequence of values).

        """
        if ix is not None and w is not None:
            return np.array(ix), np.array(w)
        else:
            print("ix and w are complementary parameters and must both be given")
            raise TypeError

    def _check_vals(self, vals):
        """TODO Basic check of target elements (sequence of polygons).

        """
        if type(self.src) is ogr.DataSource:
            lyr = self.src.GetLayerByName('src_grid')
            lyr.ResetReading()
            lyr.SetSpatialFilter(None)
            src_len = lyr.GetFeatureCount()
        else:
            src_len = len(self.src)
        assert len(vals)==src_len, "Argument vals must be of length %d" % src_len
        return vals

    def mean(self, vals):
        """
        Evaluate (weighted) zonal mean for values given at the source points.

        Parameters
        ----------
        vals : 1-d ndarray of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        self.isempty = self.check_empty()
        out = np.zeros(len(self.ix))*np.nan
        out[~self.isempty] =  np.array( [np.average( vals[self.ix[i]], weights=self.w[i] ) \
                                        for i in np.arange(len(self.ix))[~self.isempty]] )
        return out
            

    def var(self, vals):
        """
        Evaluate (weighted) zonal variance for values given at the source points.

        Parameters
        ----------
        vals : 1-d ndarray of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        mean = self.mean(vals)
        out = np.zeros(len(self.ix))*np.nan
        out[~self.isempty] = np.array( [np.average( (vals[self.ix[i]] - mean[i])**2, weights=self.w[i]) \
                                       for i in np.arange(len(self.ix))[~self.isempty]] )
        return out


class GridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------
    buffer : float (same unit as coordinates)
             Polygons will be considered inside the target if they are contained in the buffer.
    """
    def get_weights(self, trg, **kwargs):
        """
        """
        t1 = dt.datetime.now()

        # claim and reset source ogr layer
        layer = self.src.GetLayerByName('src_grid')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        areas = []
        ix = []

        trg_area = trg.Area()
        # iterate over layer features
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()
            idx = ogr_src.GetField('index')
            ix.append(idx)
            # calculate intersection, if not fully contained
            if not trg.Contains(geom):
                geom = trg.Intersection(geom)
            area = geom.Area()
            areas.append(area)

            ogr_add_feature(self.tmp_lyr, ogr_src, area / trg_area)

        areas = np.array(areas)
        w = areas / np.sum(areas)

        ix = np.array(ix)
        t2 = dt.datetime.now()
        print "Getting Weights takes: %f seconds" % (t2 - t1).total_seconds()

        return ix, w


    def _get_intersection(self, trg=None, idx=None, **kwargs):
        """Just a toy function if you want to inspect the intersection polygons of an arbitrary target
        or an target by index.
        """

        # check wether idx is given
        if idx is not None:
            if self.trg:
                try:
                    lyr = self.trg.GetLayerByName('trg_poly')
                    feat = lyr.GetFeature(idx)
                    trg = feat.GetGeometryRef()
                except:
                    raise TypeError("No target polygon found at index {0}".format(idx))
            else:
                raise TypeError('No target polygons found in object!')

        # check for trg
        if trg is None:
            raise TypeError('Either *trg* or *idx* keywords must be given!')

        # check for geometry
        if not type(trg) == ogr.Geometry:
            trg = numpy_to_ogr(trg, 'Polygon')

        # apply Buffer value
        trg = trg.Buffer(self._buffer)

        if idx is None:
            # claim and reset source layer
            # apply spatial filter
            layer = self.src.GetLayerByName('src_grid')
            layer.ResetReading()
            layer.SetSpatialFilter(trg)
        else:
            # claim and reset source layer
            layer = self._dst.GetLayerByName('dst_{0}'.format(idx))
            layer.ResetReading()

        intersecs = []
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()
            if trg.Contains(geom):
                intersecs.append(ogr_to_numpy(geom))
            else:
                # this might be wrapped in its own recursive function, with generators
                isec = trg.Intersection(geom)
                geom_name = isec.GetGeometryName()
                if geom_name in ["MULTIPOLYGON",]:
                    for i in range(isec.GetGeometryCount()):
                        intersecs.append(ogr_to_numpy(isec.GetGeometryRef(i)))
                elif isec.GetGeometryName() in ["GEOMETRYCOLLECTION"]:
                    for i in range(isec.GetGeometryCount()):
                        g = isec.GetGeometryRef(i)
                        if g.GetGeometryName() in ["POLYGON"]:
                            intersecs.append(ogr_to_numpy(g))
                elif isec.GetGeometryName() in ["POLYGON"]:
                    intersecs.append(ogr_to_numpy(isec))
                else:
                    print("Unknown Geometry:", isec.GetGeometryName(), isec.ExportToWkt())

        return np.array(intersecs)


class GridPointsToPoly(ZonalStatsBase):
    """Compute zonal average from all points in or close to the target polygon.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points points (shape Nx2) or
        ESRI Shapefile filename containing source points
    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------
    buffer : float (same unit as coordinates)
             Points will be considered inside the target if they are contained in the buffer.

    """
    def get_weights(self, trg, **kwargs):
        """
        """
        # if given, we apply a buffer value to the target polygon
        trg = trg.Buffer(self._buffer)

        ix_ = self.get_points_in_target(trg, **kwargs)
        if len(ix_) == 0:
            # No points in target polygon? Find the closest point to provide a value
            ix_ = self.get_point_next_to_target(trg, **kwargs)

        w = np.ones(len(ix_)) / len(ix_ )

        return ix_, w

    def get_points_in_target(self, trg, **kwargs):
        """Helper method that can also be used to return intermediary results.
        """
        t1 = dt.datetime.now()

        # claim and reset source ogr layer
        layer = self.src.GetLayerByName('src_grid')
        layer.ResetReading()
        layer.SetSpatialFilter(trg)

        feat_cnt = layer.GetFeatureCount()

        ix_ = [ogr_add_feature(self.tmp_lyr, ogr_src, 1. / feat_cnt) for ogr_src in layer]

        t2 = dt.datetime.now()
        print("Getting Weights takes: %f seconds" % (t2 - t1).total_seconds())

        return ix_

    def get_point_next_to_target(self, trg, dst_trg_lyr, **kwargs):
        """ Computes the target centroid and finds the closest point from src.
        """
        layer = self.src.GetLayerByName('src_grid')
        layer.ResetReading()
        layer.SetSpatialFilter(None)
        src_pts = np.array([ogr_src.GetGeometryRef().GetPoint_2D(0) for ogr_src in layer])

        centroid = get_centroid(trg)
        tree = cKDTree(src_pts)
        distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
        ogr_add_feature(dst_trg_lyr, layer.GetFeature(ixnext), 1.)
        return np.array([ixnext])


def ogr_create_datasource(drv, name, remove=False):
    """Creates OGR.DataSource object.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    drv : string, GDAL/OGR driver string

    name : string, path to filename

    remove : bool, if True, existing OGR.DataSource will be
        removed before creation

    Returns
    -------
    out : an OGR.DataSource object

    """


    drv = ogr.GetDriverByName(drv)
    if remove:
        if os.path.exists(name):
            drv.DeleteDataSource(name)
    ds = drv.CreateDataSource(name)

    return ds


def ogr_create_layer(ds, name, geom_type=None, fields=None):
    """Creates OGR.Layer objects in OGR.DataSource object.

    .. versionadded:: 0.7.0

    Creates one OGR.Layer with given name in given OGR.DataSource object
    using given OGR.GeometryType and FieldDefinitions

    Parameters
    ----------
    ds : OGR.DataSource object

    name : string, OGRLayer name

    geom_type : OGR GeometryType (eg. ogr.wkbPolygon)

    fields : dictionary, dict.keys are field names strings
            dict.values are OGR.DataTypes (eg. ogr.OFTReal)

    Returns
    -------
    out : an OGR.Layer object

    """

    if geom_type is None:
        raise TypeError("geometry_type needed")

    lyr = ds.CreateLayer(name, geom_type=geom_type)
    for fname, fvalue in fields.items():
        lyr.CreateField(ogr.FieldDefn(fname, fvalue))

    return lyr


def ogr_copy_layer(src_ds, name, dst_ds, reset=True):
    """ Copy OGR.Layer object.

    .. versionadded:: 0.7.0

    Copy OGR.Layer object from src_ds OGR.DataSource to dst_ds OGR.DataSource

    Parameters
    ----------
    src_ds : OGR.DataSource object

    name : string, name of wanted Layer

    dst_ds : OGR.DataSource object

    reset : bool, if True resets src_layer
    """
    # get and copy src geometry layer
    src_lyr = src_ds.GetLayerByName(name)
    if reset:
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
    tmp_lyr = dst_ds.CopyLayer(src_lyr, name)


def ogr_add(ds, src, name=None):
    """ Creates OGR.Feature objects in OGR.Layer object.

    .. versionadded:: 0.7.0

    OGR.Features are built from numpy src points or polygons.

    OGR.Features 'FID' and 'index' corresponds to source data element

    Parameters
    ----------
    ds : OGR.DataSource object

    src : source data numpy array

    name : string, name of wanted Layer
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

        if 'index' in fields:
            feat.SetField('index', index)

        feat.SetGeometry(geom)
        lyr.CreateFeature(feat)


def ogr_add_by_index(layer, src_layer, values):
    """ Copies OGR.Feature objects from src_layer
    to layer OGR.Layer object.

    .. versionadded:: 0.7.0

    OGR.Features are read by index from src layer. Attributes
    are attached.

    Parameters
    ----------
    layer : OGR.Layer object

    src_layer : source layer

    values : list of tupels (index, weights)
    """

    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)

    for idx, weight in values:
        src_feat = src_layer.GetFeature(idx)
        feat.SetField('index', idx)
        feat.SetField('weight', weight)
        feat.SetGeometry(src_feat.GetGeometryRef())
        layer.CreateFeature(feat)


def ogr_add_feature(layer, feature, values):
    """ Copies single OGR.Feature object to an OGR.Layer object.

    .. versionadded:: 0.7.0

    Given OGR.Feature-Geometry is copied to new OGR.Feature and
    written to given OGR.Layer by given index. Attributes are attached.

    Parameters
    ----------
    layer : OGR.Layer object

    feature : OGR.Feature Object

    values : attribute list of tupels (index, weights)
    """
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)


    weight = values
    idx = feature.GetField('index')
    feat.SetField('index', idx)
    feat.SetField('weight', weight)
    feat.SetGeometry(feature.GetGeometryRef())
    layer.CreateFeature(feat)

    return idx


def numpy_to_ogr(vert, geom_name):
    """Convert a vertex array to gdal/ogr geometry.

    .. versionadded:: 0.7.0

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    vert : a numpy array of vertices of shape (num vertices, 2)

    Returns
    -------
    out : an ogr Geometry object of type geom_name

    """

    if geom_name == 'Polygon':
        json_str = "{{'type':{0!r},'coordinates':[{1!r}]}}".format(geom_name, vert.tolist())
    else:
        json_str = "{{'type':{0!r},'coordinates':{1!r}}}".format(geom_name, vert.tolist())

    return ogr.CreateGeometryFromJson(json_str)


def ogr_to_numpy(ogrobj):
    """Backconvert a gdal/ogr geometry to a numpy vertex array.

    .. versionadded:: 0.7.0

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    ogrobsj : an ogr Geometry object

    Returns
    -------
    out : a nested ndarray of vertices of shape (num vertices, 2)

    """
    jsonobj = eval(ogrobj.ExportToJson())

    return np.squeeze(jsonobj['coordinates'])


def mask_from_bbox(x, y, bbox, polar=False):
    """Return 2-d index array based on spatial selection from a bounding box.

    Use this function to create a 2-d boolean mask from 2-d arrays of grids points.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    x : nd array of shape (num rows, num columns)
        x (Cartesian) coordinates
    y : nd array of shape (num rows, num columns)
        y (Cartesian) coordinates
    bbox : dictionary with keys "left", "right", "bottom", "top"
        These must refer to the same Cartesian reference system as x and y
    polar : x, y are aligned polar (azimuth x range)

    Returns
    -------
    out : mask, shape
          mask is a boolean array that is True if the point is inside the bbox
          shape is the shape of the True subgrid

    """
    ny, nx = x.shape

    ix = np.arange(x.size).reshape(x.shape)

    # Find bbox corners
    #    Plant a tree
    tree = cKDTree(np.vstack((x.ravel(),y.ravel())).transpose())
    # find lower left corner index
    dists, ixll = tree.query([bbox["left"], bbox["bottom"]], k=1)
    ill = (ixll / nx)-1
    jll = (ixll % nx)-1
    # find upper right corner index
    dists, ixur = tree.query([bbox["right"],bbox["top"]], k=1)
    iur = (ixur / nx)+1
    jur = (ixur % nx)+1

    # for polar grids we need all 4 corners
    if polar:
        # find upper left corner index
        dists, ixul = tree.query([bbox["left"], bbox["top"]], k=1)
        iul = (ixul / nx)-1
        jul = (ixul % nx)-1
        # find lower right corner index
        dists, ixlr = tree.query([bbox["right"],bbox["bottom"]], k=1)
        ilr = (ixlr / nx)+1
        jlr = (ixlr % nx)+1

    mask = np.repeat(False, ix.size).reshape(ix.shape)

    # for polar grids we have to handle the azimuth carefully
    if polar:
        # ranges are not problematic, just get min and max
        jmin = min(jll, jul, jur, jlr)
        jmax = max(jll, jul, jur, jlr)

        # azimuth array for angle_between calculation
        ax = np.array([[ill, ilr],
                       [ill, iur],
                       [iul, ilr],
                       [iul, iur]])

        # this calculates the angles between 4 azimuth and returns indices
        # of the greatest angle
        ar = angle_between(ax[:,0], ax[:,1])
        maxind = np.argmax(ar)
        imin, imax = ax[maxind,:]

        # if catchment extends over zero angle
        if imin > imax:
            mask[:imax, jmin:jmax] = True
            mask[imin:, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax-jmin)
        else:
            mask[imin:imax, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax-jmin)

    else:

        if iur>ill:
            mask[ill:iur,jll:jur] = True
            shape = (iur-ill, jur-jll)
        else:
            mask[iur:ill,jll:jur] = True
            shape = (ill-iur, jur-jll)

    return mask, shape


def angle_between(source_angle, target_angle):
    """Return angle between source and target radial angle
    """
    sin1 = np.sin(np.radians(target_angle)-np.radians(source_angle))
    cos1 = np.cos(np.radians(target_angle)-np.radians(source_angle))
    return np.rad2deg(np.arctan2(sin1, cos1))


def get_bbox(x, y):
    """Return bbox dictionary that represents the extent of the points.
    """
    return dict(left=np.min(x),
                right=np.max(x),
                bottom=np.min(y),
                top=np.max(y))


def get_centroid(polyg):
    """Return centroid of a polygon

    Parameters
    ----------
    polyg : ndarray of shape (num vertices, 2) or ogr.Geometry object

    Returns
    -------
    out : x and y coordinate of the centroid

    """
    if not type(polyg) == ogr.Geometry:
        polyg = numpy_to_ogr(polyg, 'Polygon')
    return polyg.Centroid().GetPoint()[0:2]


def grid_centers_to_vertices(X, Y, dx, dy):
    """Produces array of vertices from grid's center point coordinates.

    .. warning:: This has to be done in the "native" grid projection.
                 Once you reprojected the coordinates, this trivial function
                 cannot be used to compute vertices from center points.

    Parameters
    ----------
    X : 2-d array of x coordinates (same shape as the actual 2-D grid)
    Y : 2-d array of y coordinates (same shape as the actual 2-D grid)
    dx : grid spacing in x direction
    dy : grid spacing in y direction

    Returns
    -------
    out : 3-d array of vertices for each grid cell of shape (n grid points,
          5, 2)

    """
    left    = X - dx/2
    right   = X + dy/2
    bottom  = Y - dy/2
    top     = Y + dy/2

    verts = np.vstack(( [left.ravel() ,bottom.ravel()],
                        [right.ravel(),bottom.ravel()],
                        [right.ravel(),top.ravel()],
                        [left.ravel() ,top.ravel()],
                        [left.ravel() ,bottom.ravel()]) ).T.reshape((-1,5,2))

    return verts


if __name__ == '__main__':
    print 'wradlib: Calling module <zonalstats> as main...'
