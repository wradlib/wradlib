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
precipitation estimates from a radar grid in polar coordinates or from precipitation
estimates in a Cartesian grid.

The general usage is similar to the ipol and adjustment modules:

You have to create an instance of a class (derived from ZonalDataBase) by using the
spatial information of your source and target objects (e.g. radar bins and
catchment polygons). The Zonal Data within this object can be saved as an ESRI Shapefile.

This object is then called with another class to compute zonal statistics for your target
objects by calling the class instance with an array of values (one for each source object).

Typically, creating the instance of the ZonalData class will be computationally expensive,
but only has to be done once (as long as the geometries do not change).

Calling the objects with actual data, however, will be very fast.

..note:: Right now we only support a limited set of 2-dimensional zonal statistics.
         In the future, we plan to extend this to three dimensions.


.. currentmodule:: wradlib.zonalstats

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ZonalDataPoint
   ZonalDataPoly
   GridCellsToPoly
   GridPointsToPoly

"""

import os

from osgeo import gdal, ogr
import numpy as np
from scipy.spatial import cKDTree
import datetime as dt
from matplotlib.path import Path
import matplotlib.patches as patches

import wradlib.io as io


class ZonalDataBase(object):
    """
    The base class for managing 2-dimensional zonal data for target polygons
    from source points or polygons. Provides the basic design for all other classes.

    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the created destination layer will be empty.

    Data Model is built upon OGR Implementation of ESRI Shapefile

    - one src layer (named 'src_grid') holding source polygons or points
    - one trg layer (named 'trg_grid') holding target polygons
    - several dst layers (named 'dst_N', N is int number) holding src polygons/points
    related to target polygons with attached index and weights fields

    By using OGR there are no restrictions for the used source grids.

    .. versionadded:: 0.7.0

    .. warning:: Writing shapefiles with the wrong locale settings can have impact on the
    type of the decimal. If problem arise use LC_NUMERIC=C in your environment.

    Parameters
    ----------

    src : sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------

    buf : float (same unit as coordinates)
        Points will be considered inside the target if they are contained in the buffer.

    srs : OGR.SpatialReference object which will be used for Zonal Data object
        source and target data have to be in the srs-format

    """

    def __init__(self, src, trg=None, buf=0., srs=None, **kwargs):
        # if only src is given assume "dump_all_shape" filename
        self._buffer = buf
        self._srs = srs
        if trg is None:
            self.load_all_shape(src)
        else:
            self.src = self._check_src(src, **kwargs)
            self.trg = self._check_trg(trg, **kwargs)
            self.dst = self._create_dst_layers()

    @property
    def srs(self):
        """ Returns srs
        """
        return self._srs

    @property
    def isec(self):
        """ Returns intersections
        """
        raise NotImplementedError

    @property
    def targets(self):
        """ Returns target polygons
        """
        lyr = self.trg.GetLayer()
        lyr.ResetReading()
        targets = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            mpoly = patches.Polygon(poly, True)
            targets.append(mpoly)
        return targets

    @property
    def sources(self):
        """ Returns sources
        """
        raise NotImplementedError

    def get_sources(self, idx):
        """ Returns sources referring to target polygon idx
        """
        raise NotImplementedError

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
            ogr_src_lyr = ogr_src.CopyLayer(tmp_lyr, 'src_grid')
            self._geom_type = ogr_src_lyr.GetGeomType()
        except IOError:
            # no ESRI shape file
            raise
        # all failed? then it should be sequence or numpy array
        except RuntimeError:
            src = np.array(src)
            # create memory datasource, layer and create features
            if src.ndim == 3:
                self._geom_type = ogr.wkbPolygon
            # no Polygons, just Points
            else:
                self._geom_type = ogr.wkbPoint

            fields = {'index': ogr.OFTInteger}
            ogr_create_layer(ogr_src, 'src_grid', srs=self._srs, geom_type=self._geom_type, fields=fields)
            ogr_add_feature(ogr_src, src, name='src_grid')

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
            ogr_create_layer(ogr_trg, 'trg_poly', srs=self._srs, geom_type=ogr.wkbPolygon, fields=fields)
            ogr_add_feature(ogr_trg, trg, name='trg_poly')

        t2 = dt.datetime.now()
        print "Setting up Target takes: %f seconds" % (t2 - t1).total_seconds()

        return ogr_trg

    def _create_dst_layers(self, **kwargs):
        """ Create destination target OGR.DataSource

        Creates one layer for each target polygon, consisting of
        the needed source data attributed with index and weights fields

        Returns
        -------

        ds_mem : OGR.DataSource object

        """

        #TODO: kwargs necessary?

        # create intermediate mem datasource
        ds_mem = ogr_create_datasource('Memory', 'dst')

        # get src geometry layer
        src_lyr = self.src.GetLayerByName('src_grid')
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)

        fields = {'index': ogr.OFTInteger, 'weight': ogr.OFTReal}

        # get target layer, iterate over polygons and calculate weights
        lyr = self.trg.GetLayer()
        lyr.ResetReading()
        for index, trg_poly in enumerate(lyr):

            # create layer as self.tmp_lyr
            self.tmp_lyr = ogr_create_layer(ds_mem, 'dst_{0}'.format(index), srs=self._srs,
                                            geom_type=self._geom_type, fields=fields)

            # calculate weights while filling self.tmp_lyr with index and weights information
            self._create_dst_features(self.tmp_lyr, trg_poly.GetGeometryRef(), **kwargs)

        return ds_mem

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create OGR.Features in Destination OGR.Layer
        """
        raise NotImplementedError

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
        [ogr_copy_layer(self.dst, 'dst_{0}'.format(i), ds_out) for i in range(self.dst.GetLayerCount())]

        # flush everything
        del ds_out

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

    def dump_trg_shape(self, filename, remove=True):
        """ Output layer to ESRI_Shapefile

        Parameters
        ----------
        filename : string, path to shape-filename
        layer : layer to output

        """
        ds_out = ogr_create_datasource('ESRI Shapefile', filename, remove=remove)
        ogr_copy_layer(self.trg, 'trg_poly', ds_out)

        # flush everything
        del ds_out

    def dump_trg_raster(self, filename, attr, pixel_size=1., nodata=0., remove=True):
        """ Output layer to ESRI_Shapefile

        Parameters
        ----------
        filename : string, path to shape-filename
        layer : layer to output

        """
        layer = self.trg.GetLayerByIndex(0)
        layer.ResetReading()

        x_min, x_max, y_min, y_max = layer.GetExtent()

        cols = int( (x_max - x_min) / pixel_size )
        rows = int( (y_max - y_min) / pixel_size )


        drv = gdal.GetDriverByName('GTiff')
        if os.path.exists(filename):
            drv.Delete(filename)
        raster = drv.Create(filename, cols, rows, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

        band = raster.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.FlushCache()

        gdal.RasterizeLayer(raster, [1], layer, burn_values=[0], options=["ATTRIBUTE={0}".format(attr), "ALL_TOUCHED=TRUE"])
        raster.SetProjection(self._srs.ExportToWkt())

        del raster

    def load_all_shape(self, filename):
        """ Load source/target grid points/polygons into in-memory Shapefile

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
        self.trg = drv_in.CreateDataSource('trg')
        self.dst = drv_in.CreateDataSource('dst')

        # get src and target polygon layer
        ogr_copy_layer(ds_in, 'src_grid', self.src)
        ogr_copy_layer(ds_in, 'trg_poly', self.trg)

        # get destination trg layers
        [ogr_copy_layer(ds_in, 'dst_{0}'.format(i), self.dst) for i in range(ds_in.GetLayerCount() - 2)]

        self._srs = self.src.GetLayer().GetSpatialRef()

        # flush everything
        del ds_in

    def set_trg_attribute(self, name, values):

        lyr = self.trg.GetLayerByIndex(0)
        lyr.ResetReading()
        # todo: automatically check for value type
        defn = lyr.GetLayerDefn()

        if defn.GetFieldIndex(name) == -1:
            lyr.CreateField(ogr.FieldDefn(name, ogr.OFTReal))

        for i, item in enumerate(lyr):
            item.SetField(name, values[i])
            lyr.SetFeature(item)

    def _get_idx_weights(self):
        """ Read index and weights from OGR.Layer

        Parameters
        ----------
        trg_ind : int, index of target polygon

        """
        ix = []
        w = []
        for i in range(self.dst.GetLayerCount()):
            _ix, _w = self._get_idx_weights_from_layer(i)
            ix.append(_ix)
            w.append(_w)
        return ix, w

    def _get_idx_weights_from_layer(self, trg_ind):
        """ Read index and weights from OGR.Layer

        Parameters
        ----------
        trg_ind : int, index of target polygon

        """
        lyr = self.dst.GetLayerByName('dst_{0}'.format(trg_ind))
        lyr.ResetReading()
        ix = [ogr_src.GetField('index') for ogr_src in lyr]
        lyr.ResetReading()
        w = [ogr_src.GetField('weight') for ogr_src in lyr]
        return ix, w

    def _get_intersection(self, trg=None, idx=None, buf=0., **kwargs):
        """Just a toy function if you want to inspect the intersection points/polygons
        of an arbitrary target or an target by index.
        """
        #TODO: kwargs necessary?

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
        trg = trg.Buffer(buf)

        if idx is None:
            # claim and reset source layer
            # apply spatial filter
            layer = self.src.GetLayerByName('src_grid')
            layer.ResetReading()
            layer.SetSpatialFilter(trg)
        else:
            # claim and reset source layer
            layer = self.dst.GetLayerByName('dst_{0}'.format(idx))
            layer.ResetReading()
            layer.SetSpatialFilter(None)

        intersecs = []
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()
            if geom is not None:
                intersecs.append(ogr_to_numpy(geom))

        return intersecs


class ZonalDataPoly(ZonalDataBase):
    """ ZonalData object for source polygons

    .. versionadded:: 0.7.0

    Parameters
    ----------

    src : sequence of source polygons (shape NxMx2) or
        ESRI Shapefile filename containing source polygons

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------

    buf : float (same unit as coordinates)
        Polygons will be considered inside the target if they are contained
        in the buffer.

    """
    @property
    def sources(self):
        """ Returns source polygon patches

        ..note:: This may be slow, because it extracts all source polygons
        """
        lyr = self.src.GetLayer()
        sources = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            mpoly = patches.Polygon(poly, True)
            sources.append(mpoly)
        return np.array(sources)

    def get_sources(self, idx):
        """ Returns source polygon patches reffering to target polygon idx

        Parameters
        ----------
        idx : int, index of target polygon

        Returns
        -------
        array : ndarray of matplotlib.patches.Polygon objects
        """
        lyr = self.dst.GetLayerByName('dst_{0}'.format(idx))
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)

        index = [feature.GetField('index') for feature in lyr]

        lyr = self.src.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)

        sources = []
        for i in index:
            feature = lyr.GetFeature(i)
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            mpoly = patches.Polygon(poly, True)
            sources.append(mpoly)

        return np.array(sources)

    @property
    def isec(self):
        """ Returns intersections

        Returns
        -------
        array : ndarray of matplotlib.patches.PathPatch objects
        """
        tmp = [self._get_intersection(idx=idx) for idx in range(self.dst.GetLayerCount())]
        isecs = []
        for isec in tmp:
            paths = []
            for item in isec:
                if item.ndim != 2:
                    vert = np.vstack(item)
                    code = np.full(vert.shape[0], 2, dtype=np.int)
                    ind = np.cumsum([0] + [len(x) for x in item[:-1]])
                    code[ind] = 1
                    path = Path(vert, code)
                    paths.append(patches.PathPatch(path))
                else:
                    path = Path(item, [1] + (len(item)-1) * [2])
                    paths.append(patches.PathPatch(path))
            isecs.append(paths)

        return np.array(isecs)

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create needed OGR.Features in dst OGR.Layer

        Parameters
        ----------
        dst : OGR.Layer object, destination layer
        trg : OGR.Geometry object, target polygon

        """
        #TODO: kwargs necessary?

        t1 = dt.datetime.now()

        # claim and reset source ogr layer
        layer = self.src.GetLayerByName('src_grid')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        trg_area = trg.Area()
        # iterate over layer features
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()

            # calculate intersection, if not fully contained
            if not trg.Contains(geom):
                geom = trg.Intersection(geom)

            # checking GeometryCollection, convert to only Polygons, Multipolygons
            if geom.GetGeometryType() in [7]:
                geocol = ogr_geocol_to_numpy(geom)
                geom = numpy_to_ogr(geocol, 'MultiPolygon')

            # only geometries containing points
            if geom.IsEmpty():
                continue

            if geom.GetGeometryType() in [3, 6, 12]:

                idx = ogr_src.GetField('index')

                ogr_add_geometry(dst, geom, idx, geom.Area() / trg_area)

                # neccessary?
                dst.SyncToDisk()

        t2 = dt.datetime.now()
        print "Getting Weights takes: %f seconds" % (t2 - t1).total_seconds()


class ZonalDataPoint(ZonalDataBase):
    """ ZonalData object for source points

    .. versionadded:: 0.7.0

    Parameters
    ----------

    src : sequence of source points (shape Nx2) or
        ESRI Shapefile filename containing source points

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------

    buf : float (same unit as coordinates)
        Points will be considered inside the target if they are contained
        in the buffer.

    """
    @property
    def isec(self):
        """ Returns intersections

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        return [np.array(self._get_intersection(idx=idx)) for idx in range(self.dst.GetLayerCount())]

    @property
    def sources(self):
        """ Returns source points

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        lyr = self.src.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)

        sources = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            sources.append(poly)

        return np.array(sources)

    def get_sources(self, idx):
        """ Returns source points referring to target polygon idx

        Parameters
        ----------
        idx : int, index of target polygon

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        lyr = self.dst.GetLayerByName('dst_{0}'.format(idx))
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)

        index = [feature.GetField('index') for feature in lyr]

        lyr = self.src.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)

        sources = []
        for i in index:
            feature = lyr.GetFeature(i)
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            sources.append(poly)

        return np.array(sources)

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create needed OGR.Features in dst OGR.Layer

        Parameters
        ----------
        dst : OGR.Layer object, destination layer
        trg : OGR.Geometry object, target polygon

        """
        #TODO: kwargs necessary?

        t1 = dt.datetime.now()

        # claim and reset source ogr layer
        layer = self.src.GetLayerByName('src_grid')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        feat_cnt = layer.GetFeatureCount()

        if feat_cnt:
            ix_ = [ogr_add_geometry(dst, ogr_src.GetGeometryRef(),
                                    ogr_src.GetField('index'),  1. / feat_cnt) for ogr_src in layer]
        else:
            layer.SetSpatialFilter(None)
            src_pts = np.array([ogr_src.GetGeometryRef().GetPoint_2D(0) for ogr_src in layer])
            centroid = get_centroid(trg)
            tree = cKDTree(src_pts)
            distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
            feat = layer.GetFeature(ixnext)
            ogr_add_geometry(dst, feat.GetGeometryRef(), feat.GetField('index'), 1.)

        # neccessary?
        dst.SyncToDisk()

        t2 = dt.datetime.now()
        print "Getting Weights takes: %f seconds" % (t2 - t1).total_seconds()


class ZonalStatsBase(object):
    """Base class for all 2-dimensional zonal statistics.

    .. versionadded:: 0.7.0

    The base class for computing 2-dimensional zonal statistics for target
    polygons from source points or polygons as built up with ZonalDataBase
    and derived classes. Provides the basic design for all other classes.
    
    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the zonal statistic for that target will be NaN.

    Parameters
    ----------
    src : ZonalDataPoly object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary Zonal Data
        ZonalData is available as 'zdata'-property inside class instance

    """
    def __init__(self, src, **kwargs):

        self._ix = []
        self._w = []

        if isinstance(src, ZonalDataBase):
            self.zdata = src
        else:
            raise TypeError('Parameter mismatch in calling ZonalDataBase')

        self.ix, self.w = self.zdata._get_idx_weights()

    # TODO: check which properties are really needed
    @property
    def zdata(self):
        return self._zdata

    @zdata.setter
    def zdata(self, value):
        self._zdata = value

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

    def check_empty(self):
        """
        """
        isempty = np.repeat(False, len(self.w))
        for i, weights in enumerate(self.w):
            if np.sum(weights)==0 or np.isnan(np.sum(weights)):
                isempty[i] = True
        return isempty

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
        lyr = self.zdata.src.GetLayerByName('src_grid')
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        src_len = lyr.GetFeatureCount()

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

        if self.zdata is not None:
            self.zdata.set_trg_attribute('mean', out)

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

        if self.zdata is not None:
            self.zdata.set_trg_attribute('var', out)

        return out


# should we rename class?
# GridCellStats ?
class GridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : ZonalDataPoly object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary Zonal Data

    Keyword arguments
    -----------------

    """
    def __init__(self, src, **kwargs):
        if not isinstance(src, ZonalDataPoly):
            src = ZonalDataPoly(src, **kwargs)
        super(GridCellsToPoly, self).__init__(src, **kwargs)


# should we rename class?
# GridPointStats ?
class GridPointsToPoly(ZonalStatsBase):
    """Compute zonal average from all points in or close to the target polygon.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : ZonalDataPoint object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary Zonal Data

    Keyword arguments
    -----------------

    """
    def __init__(self, src, **kwargs):
        if not isinstance(src, ZonalDataPoint):
            src = ZonalDataPoint(src, **kwargs)
        super(GridPointsToPoly, self).__init__(src, **kwargs)


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


def ogr_create_layer(ds, name, srs=None, geom_type=None, fields=None):
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

    lyr = ds.CreateLayer(name, srs=srs, geom_type=geom_type)
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


def ogr_add_feature(ds, src, name=None):
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


def ogr_add_geometry(layer, geom, idx, weight):
    """ Copes single OGR.Geometry object to an OGR.Layer object.

    .. versionadded:: 0.7.0

    Given OGR.Geometry is copied to new OGR.Feature and
    written to given OGR.Layer by given index. Attributes are attached.

    Parameters
    ----------
    layer : OGR.Layer object

    geom : OGR.Geometry object

    idx : int, feature index

    weight : float, feature weight
    """
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)

    feat.SetField('index', idx)
    feat.SetField('weight', weight)
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)


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

    if geom_name in ['Polygon', 'MultiPolygon']:
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


def ogr_geocol_to_numpy(ogrobj):
    """Backconvert a gdal/ogr geometry Collection to a numpy vertex array.

    .. versionadded:: 0.7.0

    This extracts only Polygon geometries!

    Using JSON as a vehicle to efficiently deal with numpy arrays.

    Parameters
    ----------
    ogrobsj : an ogr Geometry Collection object

    Returns
    -------
    out : a nested ndarray of vertices of shape (num vertices, 2)

    """
    jsonobj = eval(ogrobj.ExportToJson())

    mpol = []
    for item in jsonobj['geometries']:
        if item['type'] == 'Polygon':
            mpol.append(item['coordinates'])

    return np.squeeze(mpol)


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
