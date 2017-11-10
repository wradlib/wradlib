#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Zonal Statistics
^^^^^^^^^^^^^^^^

.. versionadded:: 0.7.0

This module supports you in computing statistics over spatial zones. A typical
application would be to compute mean areal precipitation for a catchment by
using precipitation estimates from a radar grid in polar coordinates or from
precipitation estimates in a Cartesian grid.

The general usage is similar to the ipol and adjustment modules:

You have to create an instance of a class (derived from ZonalDataBase) by using
the spatial information of your source and target objects (e.g. radar bins and
catchment polygons). The Zonal Data within this object can be saved as an
ESRI Shapefile.

This object is then called with another class to compute zonal statistics for
your target objects by calling the class instance with an array of values
(one for each source object).

Typically, creating the instance of the ZonalData class will be computationally
expensive, but only has to be done once (as long as the geometries do
not change).

Calling the objects with actual data, however, will be very fast.

.. note:: Right now we only support a limited set of 2-dimensional zonal
         statistics. In the future, we plan to extend this to three dimensions.

.. currentmodule:: wradlib.zonalstats

.. autosummary::
   :nosignatures:
   :toctree: generated/

   DataSource
   ZonalDataBase
   ZonalDataPoint
   ZonalDataPoly
   ZonalStatsBase
   GridCellsToPoly
   GridPointsToPoly
   mask_from_bbox
   get_bbox
   grid_centers_to_vertices
   get_clip_mask

"""

import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path
import matplotlib.patches as patches
from osgeo import gdal, ogr
import wradlib.io as io
import wradlib.georef as georef
ogr.UseExceptions()
gdal.UseExceptions()


class DataSource(object):
    """ DataSource class for handling ogr/gdal vector data

    .. versionadded:: 0.7.0

    DataSource handles creates in-memory (vector) ogr DataSource object with
    one layer for point or polygon geometries.

    Parameters
    ----------
    data : sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons

    srs : ogr.SpatialReferenceSystem object
        SRS describing projection of given data

    Warning
    -------
    Writing shapefiles with the wrong locale settings can have impact on the
    type of the decimal. If problem arise use LC_NUMERIC=C in your environment.

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#\
DataSource`.
    """

    def __init__(self, data=None, srs=None, **kwargs):
        self._srs = srs
        self._name = kwargs.get('name', 'layer')
        if data is not None:
            self._ds = self._check_src(data)

    @property
    def ds(self):
        """ Returns DataSource
        """
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds = value

    @property
    def data(self):
        """ Returns DataSource geometries as numpy ndarrays

        Note
        ----
        This may be slow, because it extracts all source polygons
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        return self._get_data()

    def _get_data(self):
        """ Returns DataSource geometries as numpy ndarrays
        """
        lyr = self.ds.GetLayer()
        sources = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            poly = georef.ogr_to_numpy(geom)
            sources.append(poly)
        return np.array(sources)

    def get_data_by_idx(self, idx):
        """ Returns DataSource geometries as numpy ndarrays from given index

        Parameters
        ----------
        idx : sequence of int
            indices
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        sources = []
        for i in idx:
            feature = lyr.GetFeature(i)
            geom = feature.GetGeometryRef()
            poly = georef.ogr_to_numpy(geom)
            sources.append(poly)
        return np.array(sources)

    def get_data_by_att(self, attr=None, value=None):
        """ Returns DataSource geometries filtered by given attribute/value

        Parameters
        ----------
        attr : string
            attribute name
        value : string
            attribute value
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter("{0}={1}".format(attr, value))
        return self._get_data()

    def get_data_by_geom(self, geom=None):
        """ Returns DataSource geometries filtered by given OGR geometry

        Parameters
        ----------
        geom : OGR.Geometry object
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetAttributeFilter(None)
        lyr.SetSpatialFilter(geom)
        return self._get_data()

    def _check_src(self, src):
        """ Basic check of source elements (sequence of points or polygons).

            - array cast of source elements
            - create ogr_src datasource/layer holding src points/polygons
            - transforming source grid points/polygons to ogr.geometries
              on ogr.layer
        """
        ogr_src = io.gdal_create_dataset('Memory', 'out',
                                         gdal_type=gdal.OF_VECTOR)

        try:
            # is it ESRI Shapefile?
            ds_in, tmp_lyr = io.open_shape(src, driver=ogr.
                                           GetDriverByName('ESRI Shapefile'))
            ogr_src_lyr = ogr_src.CopyLayer(tmp_lyr, self._name)
            if self._srs is None:
                self._srs = ogr_src_lyr.GetSpatialRef()
        except IOError:
            # no ESRI shape file
            raise
        # all failed? then it should be sequence or numpy array
        except RuntimeError:
            src = np.array(src)
            # create memory datasource, layer and create features
            if src.ndim == 2:
                geom_type = ogr.wkbPoint
            # no Polygons, just Points
            else:
                geom_type = ogr.wkbPolygon
            fields = [('index', ogr.OFTInteger)]
            georef.ogr_create_layer(ogr_src, self._name, srs=self._srs,
                                    geom_type=geom_type, fields=fields)
            georef.ogr_add_feature(ogr_src, src, name=self._name)

        return ogr_src

    def dump_vector(self, filename, driver='ESRI Shapefile', remove=True):
        """ Output layer to OGR Vector File

        Parameters
        ----------
        filename : string
            path to shape-filename
        driver : string
            driver string
        remove : bool
            if True removes existing output file

        """
        ds_out = io.gdal_create_dataset(driver, filename,
                                        gdal_type=gdal.OF_VECTOR,
                                        remove=remove)
        georef.ogr_copy_layer(self.ds, 0, ds_out)

        # flush everything
        del ds_out

    def dump_raster(self, filename, driver='GTiff', attr=None,
                    pixel_size=1., remove=True):
        """ Output layer to GDAL Rasterfile

        Parameters
        ----------
        filename : string
            path to shape-filename
        driver : string
            GDAL Raster Driver
        attr : string
            attribute to burn into raster
        pixel_size : float
            pixel Size in source units
        remove : bool
            if True removes existing output file

        """
        layer = self.ds.GetLayer()
        layer.ResetReading()

        x_min, x_max, y_min, y_max = layer.GetExtent()

        cols = int((x_max - x_min) / pixel_size)
        rows = int((y_max - y_min) / pixel_size)

        # Todo: at the moment, always writing floats
        ds_out = io.gdal_create_dataset('MEM', '', cols, rows, 1,
                                        gdal_type=gdal.GDT_Float32)

        ds_out.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        proj = layer.GetSpatialRef()
        if proj is None:
            proj = self._srs
        ds_out.SetProjection(proj.ExportToWkt())

        band = ds_out.GetRasterBand(1)
        band.FlushCache()
        print("Rasterize layers")
        if attr is not None:
            gdal.RasterizeLayer(ds_out, [1], layer, burn_values=[0],
                                options=["ATTRIBUTE={0}".format(attr),
                                         "ALL_TOUCHED=TRUE"],
                                callback=gdal.TermProgress)
        else:
            gdal.RasterizeLayer(ds_out, [1], layer, burn_values=[1],
                                options=["ALL_TOUCHED=TRUE"],
                                callback=gdal.TermProgress)

        io.write_raster_dataset(filename, ds_out, driver, remove=remove)

        del ds_out

    def set_attribute(self, name, values):
        """ Add/Set given Attribute with given values

        Parameters
        ----------
        name : string
            Attribute Name
        values : :class:`numpy:numpy.ndarray`
            Values to fill in attributes
        """

        lyr = self.ds.GetLayerByIndex(0)
        lyr.ResetReading()
        # todo: automatically check for value type
        defn = lyr.GetLayerDefn()

        if defn.GetFieldIndex(name) == -1:
            lyr.CreateField(ogr.FieldDefn(name, ogr.OFTReal))

        for i, item in enumerate(lyr):
            item.SetField(name, values[i])
            lyr.SetFeature(item)

    def get_attributes(self, attrs, filt=None):
        """ Read attributes

        Parameters
        ----------
        attrs : list
            Attribute Names to retrieve
        filt : tuple
            (attname,value) for Attribute Filter
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        if filt is not None:
            lyr.SetAttributeFilter('{0}={1}'.format(*filt))
        ret = [[] for _ in attrs]
        for ogr_src in lyr:
            for i, att in enumerate(attrs):
                ret[i].append(ogr_src.GetField(att))
        return ret

    def get_geom_properties(self, props, filt=None):
        """ Read attributes

        Parameters
        ----------
        props : list
            Attribute Names to retrieve
        filt : tuple
            (attname,value) for Attribute Filter

        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        if filt is not None:
            lyr.SetAttributeFilter('{0}={1}'.format(*filt))
        ret = [[] for _ in props]
        for ogr_src in lyr:
            for i, prop in enumerate(props):
                ret[i].append(getattr(ogr_src.GetGeometryRef(), prop)())
        return ret


class ZonalDataBase(object):
    """
    The base class for managing 2-dimensional zonal data for target polygons
    from source points or polygons. Provides the basic design for all other
    classes.

    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the created destination layer will be empty.

    Data Model is built upon OGR Implementation of ESRI Shapefile

    * one src DataSource (named 'src') holding source polygons or points
    * one trg DataSource (named 'trg') holding target polygons
    * one dst DataSource (named 'dst') holding intersection polygons/points
      related to target polygons with attached index and weights fields

    By using OGR there are no restrictions for the used source grids.

    .. versionadded:: 0.7.0

    Warning
    -------
    Writing shapefiles with the wrong locale settings can have impact on the
    type of the decimal. If problem arise use LC_NUMERIC=C in your environment.

    Parameters
    ----------
    src : sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Points/Polygons  will be considered inside the target if they are
        contained in the buffer.

    srs : OGR.SpatialReference
        will be used for DataSource object.
        src and trg data have to be in the same srs-format

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.

    """
    def __init__(self, src, trg=None, buf=0., srs=None, **kwargs):
        self._buffer = buf
        self._srs = srs
        if trg is None:
            self.load_vector(src)
        else:
            self.src = DataSource(src, name='src', srs=srs, **kwargs)
            self.trg = DataSource(trg, name='trg', srs=srs, **kwargs)
            self.dst = DataSource(name='dst')
            self.dst.ds = self._create_dst_datasource()

    @property
    def srs(self):
        """ Returns SpatialReferenceSystem object
        """
        return self._srs

    @property
    def isecs(self):
        """ Returns intersections

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            of Nx2 point coordinate arrays
        """
        return np.array([self._get_intersection(idx=idx)
                         for idx in range(self.trg.ds.GetLayerByName('trg').
                                          GetFeatureCount())])

    def get_isec(self, idx):
        """ Returns intersections

        Parameters
        ----------
        idx : int
            index of target polygon

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            of Nx2 point coordinate arrays
        """
        return self._get_intersection(idx=idx)

    def get_source_index(self, idx):
        """ Returns source indices referring to target polygon idx

        Parameters
        ----------
        idx : int
            index of target polygon

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            indices
        """
        return np.array(self.dst.get_attributes(['src_index'],
                                                filt=('trg_index', idx))[0])

    def _create_dst_datasource(self, **kwargs):
        """ Create destination target gdal.Dataset

        Creates one layer for each target polygon, consisting of
        the needed source data attributed with index and weights fields

        Returns
        -------
        ds_mem : gdal.Dataset object
        """

        # TODO: kwargs necessary?

        # create intermediate mem dataset
        ds_mem = io.gdal_create_dataset('Memory', 'dst',
                                        gdal_type=gdal.OF_VECTOR)

        # get src geometry layer
        src_lyr = self.src.ds.GetLayerByName('src')
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        geom_type = src_lyr.GetGeomType()

        # create temp Buffer layer (time consuming)
        ds_tmp = io.gdal_create_dataset('Memory', 'tmp',
                                        gdal_type=gdal.OF_VECTOR)
        georef.ogr_copy_layer(self.trg.ds, 0, ds_tmp)
        tmp_trg_lyr = ds_tmp.GetLayer()

        for i in range(tmp_trg_lyr.GetFeatureCount()):
            feat = tmp_trg_lyr.GetFeature(i)
            feat.SetGeometryDirectly(feat.GetGeometryRef().
                                     Buffer(self._buffer))
            tmp_trg_lyr.SetFeature(feat)

        # get target layer, iterate over polygons and calculate intersections
        tmp_trg_lyr.ResetReading()

        self.tmp_lyr = georef.ogr_create_layer(ds_mem, 'dst', srs=self._srs,
                                               geom_type=geom_type)

        print("Calculate Intersection source/target-layers")
        try:
            tmp_trg_lyr.Intersection(src_lyr, self.tmp_lyr,
                                     options=['SKIP_FAILURES=YES',
                                              'INPUT_PREFIX=trg_',
                                              'METHOD_PREFIX=src_',
                                              'PROMOTE_TO_MULTI=YES',
                                              'PRETEST_CONTAINMENT=YES'],
                                     callback=gdal.TermProgress)
        except RuntimeError:
            # Catch RuntimeError that was reported on gdal 1.11.1
            # on Windows systems
            tmp_trg_lyr.Intersection(src_lyr, self.tmp_lyr,
                                     options=['SKIP_FAILURES=YES',
                                              'INPUT_PREFIX=trg_',
                                              'METHOD_PREFIX=src_',
                                              'PROMOTE_TO_MULTI=YES',
                                              'PRETEST_CONTAINMENT=YES'])

        return ds_mem

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create OGR.Features in Destination OGR.Layer
        """
        raise NotImplementedError

    def dump_vector(self, filename, driver='ESRI Shapefile', remove=True):
        """ Output source/target grid points/polygons to ESRI_Shapefile

        target layer features are attributed with source index and weight

        Parameters
        ----------
        filename : string
            path to shape-filename
        driver : string
            OGR Vector Driver String, defaults to 'ESRI Shapefile'
        remove : bool
            if True, existing file will be removed before creation
        """
        self.src.dump_vector(filename, driver, remove=remove)
        self.trg.dump_vector(filename, driver, remove=False)
        self.dst.dump_vector(filename, driver, remove=False)

    def load_vector(self, filename):
        """ Load source/target grid points/polygons into in-memory Shapefile

        Parameters
        ----------
        filename : string
            path to vector file
        """
        # get input file handles
        ds_in, tmp = io.open_shape(filename)

        # create all DataSources
        self.src = DataSource(name='src')
        self.src.ds = io.gdal_create_dataset('Memory', 'src',
                                             gdal_type=gdal.OF_VECTOR)
        self.trg = DataSource(name='trg')
        self.trg.ds = io.gdal_create_dataset('Memory', 'trg',
                                             gdal_type=gdal.OF_VECTOR)
        self.dst = DataSource(name='dst')
        self.dst.ds = io.gdal_create_dataset('Memory', 'dst',
                                             gdal_type=gdal.OF_VECTOR)

        # copy all layers
        georef.ogr_copy_layer_by_name(ds_in, "src", self.src.ds)
        georef.ogr_copy_layer_by_name(ds_in, "trg", self.trg.ds)
        georef.ogr_copy_layer_by_name(ds_in, "dst", self.dst.ds)

        # get spatial reference object
        self._srs = self.src.ds.GetLayer().GetSpatialRef()
        self.src._srs = self.src.ds.GetLayer().GetSpatialRef()
        self.trg._srs = self.trg.ds.GetLayer().GetSpatialRef()
        self.dst._srs = self.trg.ds.GetLayer().GetSpatialRef()

        # flush everything
        del ds_in

    def _get_idx_weights(self):
        """ Retrieve index and weight from dst DataSource
        """
        raise NotImplementedError

    def _get_intersection(self, trg=None, idx=None, buf=0.):
        """Just a toy function if you want to inspect the intersection
        points/polygons of an arbitrary target or an target by index.
        """
        # TODO: kwargs necessary?

        # check wether idx is given
        if idx is not None:
            if self.trg:
                try:
                    lyr = self.trg.ds.GetLayerByName('trg')
                    feat = lyr.GetFeature(idx)
                    trg = feat.GetGeometryRef()
                except Exception:
                    raise TypeError("No target polygon found at index {0}".
                                    format(idx))
            else:
                raise TypeError('No target polygons found in object!')

        # check for trg
        if trg is None:
            raise TypeError('Either *trg* or *idx* keywords must be given!')

        # check for geometry
        if not type(trg) == ogr.Geometry:
            trg = georef.numpy_to_ogr(trg, 'Polygon')

        # apply Buffer value
        trg = trg.Buffer(buf)

        if idx is None:
            intersecs = self.src.get_data_by_geom(trg)
        else:
            intersecs = self.dst.get_data_by_att('trg_index', idx)

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

    Keyword Arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Polygons will be considered inside the target if they are contained
        in the buffer.

    srs : OGR.SpatialReference
        will be used for DataSource object.
        src and trg data have to be in the same srs-format

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.
    """
    def _get_idx_weights(self):
        """ Retrieve index and weight from dst DataSource

        Iterates over all trg DataSource Polygons

        Returns
        -------

        ret : tuple
            (index, weight) arrays

        """
        trg = self.trg.ds.GetLayer()
        cnt = trg.GetFeatureCount()
        ret = [[] for _ in range(2)]
        for index in range(cnt):
            arr = self.dst.get_attributes(['src_index'],
                                          filt=('trg_index', index))
            w = self.dst.get_geom_properties(['Area'],
                                             filt=('trg_index', index))
            arr.append(w[0])
            for i, l in enumerate(arr):
                ret[i].append(np.array(l))
        return tuple(ret)

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create needed OGR.Features in dst OGR.Layer

        Parameters
        ----------
        dst : OGR.Layer
            destination layer
        trg : OGR.Geometry
            target polygon
        """
        # TODO: kwargs necessary?

        # claim and reset source ogr layer
        layer = self.src.ds.GetLayerByName('src')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg_index = trg.GetField('index')
        trg = trg.GetGeometryRef()
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        # iterate over layer features
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()

            # calculate intersection, if not fully contained
            if not trg.Contains(geom):
                geom = trg.Intersection(geom)

            # checking GeometryCollection, convert to only Polygons,
            #  Multipolygons
            if geom.GetGeometryType() in [7]:
                geocol = georef.ogr_geocol_to_numpy(geom)
                geom = georef.numpy_to_ogr(geocol, 'MultiPolygon')

            # only geometries containing points
            if geom.IsEmpty():
                continue

            if geom.GetGeometryType() in [3, 6, 12]:
                idx = ogr_src.GetField('index')
                georef.ogr_add_geometry(dst, geom, [idx, trg_index])


class ZonalDataPoint(ZonalDataBase):
    """ ZonalData object for source points

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points (shape Nx2) or
        ESRI Shapefile filename containing source points

    trg : sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword Arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Points will be considered inside the target if they are contained
        in the buffer.

    srs : OGR.SpatialReference
        will be used for DataSource object.
        src and trg data have to be in the same srs-format

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.
    """
    def _get_idx_weights(self):
        """ Retrieve index and weight from dst DataSource

        Iterates over all trg DataSource Polygons

        Returns
        -------
        ret : tuple
            (index, weight) arrays
        """
        trg = self.trg.ds.GetLayer()
        cnt = trg.GetFeatureCount()
        ret = [[] for _ in range(2)]
        for index in range(cnt):
            arr = self.dst.get_attributes(['src_index'],
                                          filt=('trg_index', index))
            arr.append([1. / len(arr[0])] * len(arr[0]))
            for i, l in enumerate(arr):
                ret[i].append(np.array(l))
        return tuple(ret)

    def _create_dst_features(self, dst, trg, **kwargs):
        """ Create needed OGR.Features in dst OGR.Layer

        Parameters
        ----------
        dst : OGR.Layer
            destination layer
        trg : OGR.Geometry
            target polygon
        """
        # TODO: kwargs necessary?

        # claim and reset source ogr layer
        layer = self.src.ds.GetLayerByName('src')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg_index = trg.GetField('index')
        trg = trg.GetGeometryRef()
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        feat_cnt = layer.GetFeatureCount()

        if feat_cnt:
            [georef.ogr_add_geometry(dst, ogr_src.GetGeometryRef(),
                                     [ogr_src.GetField('index'), trg_index])
             for ogr_src in layer]
        else:
            layer.SetSpatialFilter(None)
            src_pts = np.array([ogr_src.GetGeometryRef().GetPoint_2D(0)
                                for ogr_src in layer])
            centroid = georef.get_centroid(trg)
            tree = cKDTree(src_pts)
            distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
            feat = layer.GetFeature(ixnext)
            georef.ogr_add_geometry(dst, feat.GetGeometryRef(),
                                    [feat.GetField('index'), trg_index])


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
    src : ZonalDataPoly
        object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary ZonalData
        ZonalData is available as 'zdata'-property inside class instance.

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#\
ZonalStats`.
    """
    def __init__(self, src=None, ix=None, w=None):

        self._ix = None
        self._w = None

        if src is not None:
            if isinstance(src, ZonalDataBase):
                self._zdata = src
            else:
                raise TypeError('Parameter mismatch in calling ZonalDataBase')
            self.ix, self.w = self._check_ix_w(*self.zdata._get_idx_weights())
        else:
            self._zdata = None
            self.ix, self.w = self._check_ix_w(ix, w)

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
            if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
                isempty[i] = True
        return isempty

    def _check_ix_w(self, ix, w):
        """TODO Basic check of target attributes (sequence of values).

        """
        if ix is not None and w is not None:
            if len(ix) != len(w):
                raise TypeError("parameters ix and w must be of equal length")
            return np.array(ix), np.array(w)
        else:
            print("ix and w are complementary parameters and "
                  "must both be given")
            raise TypeError

    def _check_vals(self, vals):
        """TODO Basic check of target elements (sequence of polygons).

        """
        if self.zdata is not None:
            lyr = self.zdata.src.ds.GetLayerByName('src')
            lyr.ResetReading()
            lyr.SetSpatialFilter(None)
            src_len = lyr.GetFeatureCount()
            assert len(vals) == src_len, \
                "Argument vals must be of length %d" % src_len
        else:
            imax = 0
            for i in self.ix:
                mx = np.nanmax(i)
                if imax < mx:
                    imax = mx
            assert len(vals) > imax, \
                "Argument vals cannot be subscripted by given index values"

        return vals

    def mean(self, vals):
        """
        Evaluate (weighted) zonal mean for values given at the source points.

        Parameters
        ----------
        vals : 1-d :class:`numpy:numpy.ndarray`
            of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        self.isempty = self.check_empty()
        out = np.zeros(len(self.ix)) * np.nan
        out[~self.isempty] = np.array(
            [np.average(vals[self.ix[i]], weights=self.w[i])
             for i in np.arange(len(self.ix))[~self.isempty]])

        if self.zdata is not None:
            self.zdata.trg.set_attribute('mean', out)

        return out

    def var(self, vals):
        """
        Evaluate (weighted) zonal variance for values given at the source
        points.

        Parameters
        ----------
        vals : 1-d :class:`numpy:numpy.ndarray`
            of type float with the same length as self.src
            Values at the source element for which to compute
            zonal statistics

        """
        self._check_vals(vals)
        mean = self.mean(vals)
        out = np.zeros(len(self.ix)) * np.nan
        out[~self.isempty] = np.array(
            [np.average((vals[self.ix[i]] - mean[i])**2, weights=self.w[i])
             for i in np.arange(len(self.ix))[~self.isempty]])

        if self.zdata is not None:
            self.zdata.trg.set_attribute('var', out)

        return out


# should we rename class?
# GridCellStats ?
class GridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : ZonalDataPoly
        object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary Zonal Data

    Keyword arguments
    -----------------

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalStats`
    and :ref:`notebooks/zonalstats/wradlib_zonalstats_example.ipynb`.
    """
    def __init__(self, src=None, **kwargs):
        if src is not None:
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
    src : ZonalDataPoint
        object or filename pointing to ZonalDataPoly ESRI shapefile
        containing necessary Zonal Data

    Keyword arguments
    -----------------

    Examples
    --------
    See :ref:`notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalStats`
    and :ref:`notebooks/zonalstats/wradlib_zonalstats_example.ipynb`.
    """
    def __init__(self, src, **kwargs):
        if src is not None:
            if not isinstance(src, ZonalDataPoint):
                src = ZonalDataPoint(src, **kwargs)
        super(GridPointsToPoly, self).__init__(src, **kwargs)


def numpy_to_pathpatch(arr):
    """ Returns PathPatches from nested array

    Parameters
    ----------
    arr : :class:`numpy:numpy.ndarray`
        numpy array of Polygon/Multipolygon vertices

    Returns
    -------
    array : :class:`numpy:numpy.ndarray`
        of matplotlib.patches.PathPatch objects
    """
    paths = []
    for item in arr:
        if item.ndim != 2:
            vert = np.vstack(item)
            code = np.full(vert.shape[0], 2, dtype=np.int)
            ind = np.cumsum([0] + [len(x) for x in item[:-1]])
            code[ind] = 1
            path = Path(vert, code)
            paths.append(patches.PathPatch(path))
        else:
            path = Path(item, [1] + (len(item) - 1) * [2])
            paths.append(patches.PathPatch(path))

    return np.array(paths)


def mask_from_bbox(x, y, bbox, polar=False):
    """Return 2-d index array based on spatial selection from a bounding box.

    Use this function to create a 2-d boolean mask from 2-d arrays of grids
    points.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        of shape (num rows, num columns)
        x (Cartesian) coordinates
    y : :class:`numpy:numpy.ndarray`
        of shape (num rows, num columns)
        y (Cartesian) coordinates
    bbox : dict
        dictionary with keys "left", "right", "bottom", "top"
        These must refer to the same Cartesian reference system as x and y
    polar : bool
        if True, x, y are aligned polar (azimuth x range)

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
    tree = cKDTree(np.vstack((x.ravel(), y.ravel())).transpose())
    # find lower left corner index
    dists, ixll = tree.query([bbox["left"], bbox["bottom"]], k=1)
    ill = (ixll // nx) - 1
    jll = (ixll % nx) - 1
    # find upper right corner index
    dists, ixur = tree.query([bbox["right"], bbox["top"]], k=1)
    iur = int(ixur / nx) + 1
    jur = (ixur % nx) + 1

    # for polar grids we need all 4 corners
    if polar:
        # find upper left corner index
        dists, ixul = tree.query([bbox["left"], bbox["top"]], k=1)
        iul = (ixul // nx) - 1
        jul = (ixul % nx) - 1
        # find lower right corner index
        dists, ixlr = tree.query([bbox["right"], bbox["bottom"]], k=1)
        ilr = (ixlr // nx) + 1
        jlr = (ixlr % nx) + 1

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
                       [iul, iur]], dtype=int)
        # this calculates the angles between 4 azimuth and returns indices
        # of the greatest angle
        ar = angle_between(ax[:, 0], ax[:, 1])
        maxind = int(np.argmax(ar))
        imin, imax = ax[maxind, :]

        # if catchment extends over zero angle
        if imin > imax:
            mask[:imax, jmin:jmax] = True
            mask[imin:, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax - jmin)
        else:
            mask[imin:imax, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax - jmin)

    else:

        if iur > ill:
            mask[ill:iur, jll:jur] = True
            shape = (iur - ill, jur - jll)
        else:
            mask[iur:ill, jll:jur] = True
            shape = (ill - iur, jur - jll)

    return mask, shape


def angle_between(source_angle, target_angle):
    """Return angle between source and target radial angle

    Parameters
    ----------
    source_angle : starting angle
    target_angle : target angle
    """
    sin1 = np.sin(np.radians(target_angle) - np.radians(source_angle))
    cos1 = np.cos(np.radians(target_angle) - np.radians(source_angle))
    return np.rad2deg(np.arctan2(sin1, cos1))


def get_bbox(x, y):
    """Return bbox dictionary that represents the extent of the points.

    Parameters
    ----------

    x : :class:`numpy:numpy.ndarray`
        x-coordinate values
    y : :class:`numpy:numpy.ndarray`
        y-coordinate values

    """
    return dict(left=np.min(x),
                right=np.max(x),
                bottom=np.min(y),
                top=np.max(y))


def grid_centers_to_vertices(x, y, dx, dy):
    """Produces array of vertices from grid's center point coordinates.

    Warning
    -------
    This has to be done in the "native" grid projection.
    Once you reprojected the coordinates, this trivial function cannot be used
    to compute vertices from center points.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        2-d array of x coordinates (same shape as the actual 2-D grid)
    y : :class:`numpy:numpy.ndarray`
        2-d array of y coordinates (same shape as the actual 2-D grid)
    dx : grid spacing in x direction
    dy : grid spacing in y direction

    Returns
    -------
    out : 3-d array
        of vertices for each grid cell of shape (n grid points,5, 2)
    """
    top = y + dy / 2
    left = x - dx / 2
    right = x + dy / 2
    bottom = y - dy / 2

    verts = np.vstack(([left.ravel(), bottom.ravel()],
                       [right.ravel(), bottom.ravel()],
                       [right.ravel(), top.ravel()],
                       [left.ravel(), top.ravel()],
                       [left.ravel(), bottom.ravel()])).T.reshape((-1, 5, 2))

    return verts


def get_clip_mask(coords, clippoly, srs):
    """Returns boolean mask of points (coords) located inside polygon clippoly

    .. versionadded:: 0.10.0

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        array of xy coords with shape [...,2]

    clippoly : :class:`numpy:numpy.ndarray`
        array of xy coords with shape (N,2) representing closed
        polygon coordinates

    srs: osr.SpatialReference

    Returns
    -------
    src_mask : :class:`numpy:numpy.ndarray`
        boolean array of shape coords.shape[0:-1]

    """
    clip = [clippoly]

    zd = ZonalDataPoint(coords.reshape(-1, coords.shape[-1]),
                        clip, srs=srs)
    obj = GridPointsToPoly(zd)

    #    Get source indices within polygon from zonal object
    #    (0 because we have only one zone)
    pr_idx = obj.zdata.get_source_index(0)

    # Subsetting in order to use only precipitating profiles
    src_mask = np.zeros(coords.shape[0:-1], dtype=np.bool)
    mask = np.unravel_index(pr_idx, coords.shape[0:-1])
    src_mask[mask] = True

    return src_mask


if __name__ == '__main__':
    print('wradlib: Calling module <zonalstats> as main...')
