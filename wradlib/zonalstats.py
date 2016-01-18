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

Calling the objects with actual data, however, will be very fast.Fast

.. note:: Right now we only support a limited set of 2-dimensional zonal statistics.
         In the future, we plan to extend this to three dimensions.

.. currentmodule:: wradlib.zonalstats

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ZonalDataBase
   ZonalDataPoint
   ZonalDataPoly
   ZonalStatsBase
   GridCellsToPoly
   GridPointsToPoly

"""

import os

from osgeo import gdal, ogr
ogr.UseExceptions()
import numpy as np
from scipy.spatial import cKDTree
import datetime as dt
from matplotlib.path import Path
import matplotlib.patches as patches


import wradlib.io as io


class DataSource(object):

    def __init__(self, data=None, srs=None, **kwargs):
        self._srs = srs
        self._name = kwargs.get('name', 'layer')
        if data is not None:
            self._ds = self._check_src(data, **kwargs)

    @property
    def ds(self):
        """ Returns ds
        """
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds = value

    @property
    def data(self):
        """ Returns DataSource geometries as numpy ndarrays

        ..note:: This may be slow, because it extracts all source polygons
        """
        return self.get_data_by_idx()

    def get_data_by_idx(self, idx=None):
        """ Returns DataSource geometries as numpy ndarrays by their index
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        sources = []
        if idx is None:
            for feature in lyr:
                geom = feature.GetGeometryRef()
                poly = ogr_to_numpy(geom)
                sources.append(poly)
        else:
            for i in idx:
                feature = lyr.GetFeature(i)
                geom = feature.GetGeometryRef()
                poly = ogr_to_numpy(geom)
                sources.append(poly)

        return np.array(sources)

    def get_data_by_att(self, attr=None, value=None):
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter("{0}={1}".format(attr, value))
        sources = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            poly = ogr_to_numpy(geom)
            sources.append(poly)
        return np.array(sources)

    def _check_src(self, src, **kwargs):
        """ Basic check of source elements (sequence of points or polygons).

            - array cast of source elements
            - create ogr_src datasource/layer holding src points/polygons
            - transforming source grid points/polygons to ogr.geometries on ogr.layer

        """
        t1 = dt.datetime.now()

        ogr_src = ogr_create_datasource('Memory', 'out')

        try:
            # is it ESRI Shapefile?
            ds_in, tmp_lyr = io.open_shape(src, driver=ogr.GetDriverByName('ESRI Shapefile'))
            ogr_src_lyr = ogr_src.CopyLayer(tmp_lyr, self._name)
            self._srs = ogr_src_lyr.GetSpatialRef()
            #self._geom_type = ogr_src_lyr.GetGeomType()
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
            ogr_create_layer(ogr_src, self._name, srs=self._srs, geom_type=geom_type, fields=fields)
            ogr_add_feature(ogr_src, src, name=self._name)

        t2 = dt.datetime.now()
        print "Setting up Data OGR Layer takes: %f seconds" % (t2 - t1).total_seconds()

        return ogr_src

    def dump_vector(self, filename, driver='ESRI Shapefile', remove=True):
        """ Output layer to ESRI_Shapefile

        Parameters
        ----------
        filename : string, path to shape-filename
        driver : string, driver string

        """
        lc = self.ds.GetLayerCount()
        root, ext = os.path.splitext(filename)

        ds_out = ogr_create_datasource(driver, filename, remove=remove)

        for i in range(lc):
            try:
                ogr_copy_layer(self.ds, i, ds_out)
            except:
                del ds_out
                ds_out = ogr_create_datasource(driver, '{0}_{1}{2}'.format(root,i,ext), remove=remove)
                ogr_copy_layer(self.ds, i, ds_out)

        # flush everything
        del ds_out

    def dump_raster(self, filename, driver, attr, pixel_size=1., nodata=0., remove=True):
        """ Output layer to GDAL Rasterfile

        Parameters
        ----------
        filename : string, path to shape-filename
        layer : layer to output

        """
        lc = self.ds.GetLayerCount()
        root, ext = os.path.splitext(filename)

        layer = self.ds.GetLayerByIndex(0)
        layer.ResetReading()

        x_min, x_max, y_min, y_max = layer.GetExtent()

        cols = int( (x_max - x_min) / pixel_size )
        rows = int( (y_max - y_min) / pixel_size )

        ds_out = gdal_create_dataset(driver, filename, cols, rows, gdal.GDT_Float32, remove=remove)
        ds_out.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

        for i in range(lc):
            try:
                band = ds_out.GetRasterBand(1)
                band.SetNoDataValue(nodata)
                band.FlushCache()
                gdal.RasterizeLayer(ds_out, [1], layer, burn_values=[0], options=["ATTRIBUTE={0}".format(attr), "ALL_TOUCHED=TRUE"])
                ds_out.SetProjection(layer.GetSpatialRef().ExportToWkt())

            except:
                del ds_out
                ds_out = gdal_create_dataset(driver, '{0}_{1}{2}'.format(root,i,ext), cols, rows, gdal.GDT_Float32, remove=remove)
                ds_out.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
                band = ds_out.GetRasterBand(1)
                band.SetNoDataValue(nodata)
                band.FlushCache()
                gdal.RasterizeLayer(ds_out, [1], layer, burn_values=[0], options=["ATTRIBUTE={0}".format(attr), "ALL_TOUCHED=TRUE"])
                ds_out.SetProjection(layer.GetSpatialRef().ExportToWkt())

        del ds_out

    def set_attribute(self, name, values):

        lyr = self.ds.GetLayerByIndex(0)
        lyr.ResetReading()
        # todo: automatically check for value type
        defn = lyr.GetLayerDefn()

        if defn.GetFieldIndex(name) == -1:
            lyr.CreateField(ogr.FieldDefn(name, ogr.OFTReal))

        for i, item in enumerate(lyr):
            item.SetField(name, values[i])
            lyr.SetFeature(item)


class ZonalDataBase(object):
    """
    The base class for managing 2-dimensional zonal data for target polygons
    from source points or polygons. Provides the basic design for all other classes.

    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the created destination layer will be empty.

    Data Model is built upon OGR Implementation of ESRI Shapefile

    * one src layer (named 'src_grid') holding source polygons or points
    * one trg layer (named 'trg_grid') holding target polygons
    * several dst layers (named 'dst_N', N is int number) holding src polygons/points
    related to target polygons with attached index and weights fields

    By using OGR there are no restrictions for the used source grids.

    .. versionadded:: 0.7.0

    .. warning::

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
            print("Load_all_shape")
            self.load_vector(src)
        else:
            print("create from scratch")
            self.src = DataSource(src, name='src', srs=srs, **kwargs)
            self.trg = DataSource(trg, name='trg', srs=srs, **kwargs)
            self.dst = DataSource(name='dst')
            self.dst.ds = self._create_dst_layers()

    @property
    def srs(self):
        """ Returns srs
        """
        return self._srs

    @property
    def isecs(self):
        """ Returns intersections

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        #trg = self.trg.ds.GetLayerByName('trg').GetFeatureCount()
        return np.array([self._get_intersection(idx=idx)
                         for idx in range(self.trg.ds.GetLayerByName('trg').GetFeatureCount())])

    @property
    def sources(self):
        """ Returns sources
        """
        raise NotImplementedError

    def get_sources(self, idx):
        """ Returns sources referring to target polygon idx
        """
        raise NotImplementedError

    def get_isec(self, idx):
        """ Returns intersections

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        #trg = self.trg.ds.GetLayerByName('trg').GetFeatureCount()
        return self._get_intersection(idx=idx)

    def get_source_index(self, idx):
        """ Returns source polygon patches referring to target polygon idx

        Parameters
        ----------
        idx : int, index of target polygon

        Returns
        -------
        array : ndarray of matplotlib.patches.Polygon objects
        """
        lyr = self.dst.ds.GetLayerByName('dst')
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter("trg={0}".format(idx))

        index = [feature.GetField('src') for feature in lyr]

        return np.array(index)

    # def _check_src(self, src, **kwargs):
    #     """ Basic check of source elements (sequence of points or polygons).
    #
    #         - array cast of source elements
    #         - create ogr_src datasource/layer holding src points/polygons
    #         - transforming source grid points/polygons to ogr.geometries on ogr.layer
    #
    #     """
    #     t1 = dt.datetime.now()
    #
    #     ogr_src = ogr_create_datasource('Memory', 'src')
    #
    #     try:
    #         # is it ESRI Shapefile?
    #         ds_in, tmp_lyr = io.open_shape(src, driver=ogr.GetDriverByName('ESRI Shapefile'))
    #         ogr_src_lyr = ogr_src.CopyLayer(tmp_lyr, 'src_grid')
    #         #self._geom_type = ogr_src_lyr.GetGeomType()
    #     except IOError:
    #         # no ESRI shape file
    #         raise
    #     # all failed? then it should be sequence or numpy array
    #     except RuntimeError:
    #         src = np.array(src)
    #         # create memory datasource, layer and create features
    #         if src.ndim == 3:
    #             #self._geom_type = ogr.wkbPolygon
    #             geom_type = ogr.wkbPolygon
    #         # no Polygons, just Points
    #         else:
    #             #self._geom_type = ogr.wkbPoint
    #             geom_type = ogr.wkbPoint
    #
    #         fields = {'index': ogr.OFTInteger}
    #         ogr_create_layer(ogr_src, 'src_grid', srs=self._srs, geom_type=geom_type, fields=fields)
    #         ogr_add_feature(ogr_src, src, name='src_grid')
    #
    #     t2 = dt.datetime.now()
    #     print "Setting up OGR Layer takes: %f seconds" % (t2 - t1).total_seconds()
    #
    #     return ogr_src
    #
    # def _check_trg(self, trg, **kwargs):
    #     """ Basic check of target elements (sequence of points or polygons).
    #
    #         Iterates over target elements (and transforms to ogr.Polygon if necessary)
    #         create ogr_trg datasource/layer holding target polygons
    #
    #     """
    #     t1 = dt.datetime.now()
    #     # if no targets are given
    #
    #     # create target polygon ogr.DataSource with dedicated target polygon layer
    #     ogr_trg = ogr_create_datasource('Memory', 'trg')
    #
    #     try:
    #         # is it ESRI Shapefile?
    #         ds, tmp_lyr = io.open_shape(trg, driver=ogr.GetDriverByName('ESRI Shapefile'))
    #         ogr_trg_lyr = ogr_trg.CopyLayer(tmp_lyr, 'trg_poly')
    #     except IOError:
    #         # no ESRI shape file
    #         raise
    #     except RuntimeError:
    #         trg = np.array(trg)
    #         # create layer and features
    #         fields = {'index': ogr.OFTInteger}
    #         ogr_create_layer(ogr_trg, 'trg_poly', srs=self._srs, geom_type=ogr.wkbPolygon, fields=fields)
    #         ogr_add_feature(ogr_trg, trg, name='trg_poly')
    #
    #     t2 = dt.datetime.now()
    #     print "Setting up Target takes: %f seconds" % (t2 - t1).total_seconds()
    #
    #     return ogr_trg

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
        src_lyr = self.src.ds.GetLayerByName('src')
        src_lyr.ResetReading()
        #src_lyr.SetSpatialFilter(None)
        geom_type = src_lyr.GetGeomType()

        #fields = {'index': ogr.OFTInteger, 'trg_index': ogr.OFTInteger, 'weight': ogr.OFTReal}
        #fields = {'src': ogr.OFTInteger, 'trg': ogr.OFTInteger, 'weight': ogr.OFTReal}
        fields = [('src', ogr.OFTInteger), ('trg', ogr.OFTInteger), ('weight', ogr.OFTReal)]

        # get target layer, iterate over polygons and calculate weights
        lyr = self.trg.ds.GetLayerByName('trg')
        lyr.ResetReading()


        #newGeometry = None
        multi = None#ogr.Geometry(ogr.wkbMultiPolygon)
        for i, feature in enumerate(lyr):
            geometry = feature.GetGeometryRef()
            if multi is None:
                multi = geometry.Clone()
            else:
                multi = multi.Union(geometry.Clone())
                print(multi.Centroid())

        #mc = multi.UnionCascaded()
        print(multi.GetGeometryType())
        print(multi.GetGeometryName())
        print(multi.GetGeometryCount())
        #print(multi)
        print(dir(multi))

        #src_lyr.SetSpatialFilter(multi)

        lyr.ResetReading()
        t1 = dt.datetime.now()
        if 0:
            self.tmp_lyr = ogr_create_layer(ds_mem, 'dst_{0}'.format(0), srs=self._srs,
                                            geom_type=geom_type)#, fields=fields)

            lyr.Intersection(src_lyr, self.tmp_lyr, options=['INPUT_PREFIX=trg_', 'METHOD_PREFIX=src_', 'PROMOTE_TO_MULTI=YES'])
        else:
            self.tmp_lyr = ogr_create_layer(ds_mem, 'dst', srs=self._srs,
                                            geom_type=geom_type, fields=fields)
            for index, trg_poly in enumerate(lyr):

                # create layer as self.tmp_lyr
                #self.tmp_lyr = ogr_create_layer(ds_mem, 'dst_{0}'.format(index), srs=self._srs,
                #                                geom_type=geom_type, fields=fields)

                # calculate weights while filling self.tmp_lyr with index and weights information
                #self._create_dst_features(self.tmp_lyr, trg_poly.GetGeometryRef(), **kwargs)
                self._create_dst_features(self.tmp_lyr, trg_poly, **kwargs)

        t2 = dt.datetime.now()
        print "Getting Intersection takes: %f seconds" % (t2 - t1).total_seconds()

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
        driver : string, OGR Vector Driver String

        filename : string, path to shape-filename

        """
        self.src.dump_vector(filename, driver, remove=remove)
        self.trg.dump_vector(filename, driver, remove=False)
        self.dst.dump_vector(filename, driver, remove=False)

    # def dump_src_shape(self, filename, remove=True):
    #     """ Output source grid points/polygons to ESRI_Shapefile
    #
    #     Parameters
    #     ----------
    #     filename : string, path to shape-filename
    #
    #     """
    #     ds_out = ogr_create_datasource('ESRI Shapefile', filename, remove=remove)
    #     #ogr_copy_layer(self.src.ds, 'src_grid', ds_out)
    #     ogr_copy_layer(self.src.ds, 0, ds_out)
    #
    #     # flush everything
    #     del ds_out

    # def dump_trg_vector(self, filename, driver, remove=True):
    #     """ Output layer to ESRI_Shapefile
    #
    #     Parameters
    #     ----------
    #     filename : string, path to shape-filename
    #     driver : string, driver string
    #
    #     """
    #     lc = self.trg.ds.GetLayerCount()
    #     root, ext = os.path.splitext(filename)
    #
    #     ds_out = ogr_create_datasource(driver, filename, remove=remove)
    #
    #     for i in range(lc):
    #         try:
    #             ogr_copy_layer(self.trg.ds, i, ds_out)
    #         except:
    #             del ds_out
    #             ds_out = ogr_create_datasource(driver, '{0}_{1}{2}'.format(root,i,ext), remove=True)
    #             ogr_copy_layer(self.trg.ds, i, ds_out)
    #
    #     # flush everything
    #     del ds_out

    # def dump_trg_raster(self, filename, attr, pixel_size=1., nodata=0., remove=True):
    #     """ Output layer to ESRI_Shapefile
    #
    #     Parameters
    #     ----------
    #     filename : string, path to shape-filename
    #     layer : layer to output
    #
    #     """
    #     layer = self.trg.ds.GetLayerByIndex(0)
    #     layer.ResetReading()
    #
    #     x_min, x_max, y_min, y_max = layer.GetExtent()
    #
    #     cols = int( (x_max - x_min) / pixel_size )
    #     rows = int( (y_max - y_min) / pixel_size )
    #
    #     drv = gdal.GetDriverByName('GTiff')
    #     if os.path.exists(filename):
    #         drv.Delete(filename)
    #     raster = drv.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    #     raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    #
    #     band = raster.GetRasterBand(1)
    #     band.SetNoDataValue(nodata)
    #     band.FlushCache()
    #
    #     gdal.RasterizeLayer(raster, [1], layer, burn_values=[0], options=["ATTRIBUTE={0}".format(attr), "ALL_TOUCHED=TRUE"])
    #     raster.SetProjection(self._srs.ExportToWkt())
    #
    #    del raster

    def load_vector(self, filename):
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
        self.src = DataSource(name='src')
        self.src.ds = drv_in.CreateDataSource('src')
        self.trg = DataSource(name='trg')
        self.trg.ds = drv_in.CreateDataSource('trg')
        self.dst = DataSource(name='trg')
        self.dst.ds = drv_in.CreateDataSource('dst')

        # copy all layers
        ogr_copy_layer(ds_in, 0, self.src.ds)
        ogr_copy_layer(ds_in, 1, self.trg.ds)
        ogr_copy_layer(ds_in, 2, self.dst.ds)

        # get spatial reference object
        self._srs = self.src.ds.GetLayer().GetSpatialRef()
        self.src._srs = self.src.ds.GetLayer().GetSpatialRef()
        self.trg._srs = self.trg.ds.GetLayer().GetSpatialRef()
        self.dst._srs = self.trg.ds.GetLayer().GetSpatialRef()

        # flush everything
        del ds_in

    # def set_trg_attribute(self, name, values):
    #
    #     lyr = self.trg.ds.GetLayerByIndex(0)
    #     lyr.ResetReading()
    #     # todo: automatically check for value type
    #     defn = lyr.GetLayerDefn()
    #
    #     if defn.GetFieldIndex(name) == -1:
    #         lyr.CreateField(ogr.FieldDefn(name, ogr.OFTReal))
    #
    #     for i, item in enumerate(lyr):
    #         item.SetField(name, values[i])
    #         lyr.SetFeature(item)

    def _get_idx_weights(self):
        """ Read index and weights from OGR.Layer

        Parameters
        ----------
        trg_ind : int, index of target polygon

        """
        ix = []
        w = []
        lyr = self.dst.ds.GetLayer()
        trg = self.trg.ds.GetLayer()
        cnt = trg.GetFeatureCount()
        for index in range(cnt):
            lyr.ResetReading()
            lyr.SetAttributeFilter('trg={0}'.format(index))
            _ix = np.array([ogr_src.GetField('src') for ogr_src in lyr])
            lyr.ResetReading()
            #_w = [ogr_src.GetField('weight') for ogr_src in lyr]
            _w = np.array([ogr_src.GetGeometryRef().Area() for ogr_src in lyr]) / len(_ix)
            ix.append(_ix)
            w.append(_w)
        return ix, w

    def _get_idx_weights_from_layer(self, trg_ind):
        """ Read index and weights from OGR.Layer

        Parameters
        ----------
        trg_ind : int, index of target polygon

        """
        lyr = self.dst.ds.GetLayerByName('dst_{0}'.format(trg_ind))
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
                    lyr = self.trg.ds.GetLayerByName('trg')
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
            layer = self.src.ds.GetLayerByName('src')
            layer.ResetReading()
            layer.SetSpatialFilter(trg)
        else:
            # claim and reset source layer
            layer = self.dst.ds.GetLayerByName('dst')
            layer.ResetReading()
            layer.SetAttributeFilter('trg={0}'.format(idx))

        intersecs = []
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()
            if geom is not None:
                intersecs.append(ogr_to_numpy(geom))

        return np.array(intersecs)


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
    # def get_source_index(self, idx):
    #     """ Returns source polygon patches referring to target polygon idx
    #
    #     Parameters
    #     ----------
    #     idx : int, index of target polygon
    #
    #     Returns
    #     -------
    #     array : ndarray of matplotlib.patches.Polygon objects
    #     """
    #     lyr = self.dst.ds.GetLayerByName('dst')
    #     lyr.ResetReading()
    #     lyr.SetSpatialFilter(None)
    #     lyr.SetAttributeFilter("trg={0}".format(idx))
    #
    #     index = [feature.GetField('src') for feature in lyr]
    #
    #     return np.array(index)

    # def get_sources(self, idx):
    #     """ Returns source polygon patches referring to target polygon idx
    #
    #     Parameters
    #     ----------
    #     idx : int, index of target polygon
    #
    #     Returns
    #     -------
    #     array : ndarray of matplotlib.patches.Polygon objects
    #     """
    #     lyr = self.dst.ds.GetLayerByName('dst')
    #     lyr.ResetReading()
    #     lyr.SetSpatialFilter(None)
    #     lyr.SetAttributeFilter("trg={0}".format(idx))
    #
    #     index = [feature.GetField('src') for feature in lyr]
    #
    #     lyr = self.src.ds.GetLayer()
    #     lyr.ResetReading()
    #     lyr.SetSpatialFilter(None)
    #
    #     sources = []
    #     for i in index:
    #         feature = lyr.GetFeature(i)
    #         geom = feature.GetGeometryRef()
    #         poly = ogr_to_numpy(geom)
    #         sources.append(poly)
    #
    #     return np.array(sources)

    # @property
    # def isec1(self):
    #     """ Returns intersections
    #
    #     Returns
    #     -------
    #     array : ndarray of matplotlib.patches.PathPatch objects
    #     """
    #     tmp = [self._get_intersection(idx=idx) for idx in range(self.dst.ds.GetLayerCount())]
    #     isecs = []
    #     for isec in tmp:
    #         paths = []
    #         for item in isec:
    #             if item.ndim != 2:
    #                 vert = np.vstack(item)
    #                 code = np.full(vert.shape[0], 2, dtype=np.int)
    #                 ind = np.cumsum([0] + [len(x) for x in item[:-1]])
    #                 code[ind] = 1
    #                 path = Path(vert, code)
    #                 paths.append(patches.PathPatch(path))
    #             else:
    #                 path = Path(item, [1] + (len(item)-1) * [2])
    #                 paths.append(patches.PathPatch(path))
    #         isecs.append(paths)
    #
    #     return np.array(isecs)

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
        layer = self.src.ds.GetLayerByName('src')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        trg_index = trg.GetField('index')
        trg = trg.GetGeometryRef()
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        trg_area = trg.Area()
        # iterate over layer features
        acc = (t1 - t1)
        acc1 = (t1 - t1)
        acc2 = (t1 - t1)
        for ogr_src in layer:
            geom = ogr_src.GetGeometryRef()

            # calculate intersection, if not fully contained
            #t3 = dt.datetime.now()
            contains = trg.Contains(geom)
            #acc = (acc + (dt.datetime.now() -t3))

            if not contains:
            #if not geom.Within(trg):
             #   t4 = dt.datetime.now()
                geom = trg.Intersection(geom)
             #   acc1 = (acc1 + (dt.datetime.now() -t4))
                #geom = geom.Intersection(trg)
            #else:
            #    t5 = dt.datetime.now()
            #    geom = trg.Intersection(geom)
            #    acc2 = (acc2 + (dt.datetime.now() -t5))

            # checking GeometryCollection, convert to only Polygons, Multipolygons
            if geom.GetGeometryType() in [7]:
                geocol = ogr_geocol_to_numpy(geom)
                geom = numpy_to_ogr(geocol, 'MultiPolygon')

            # only geometries containing points
            if geom.IsEmpty():
                continue

            if geom.GetGeometryType() in [3, 6, 12]:

                idx = ogr_src.GetField('index')

                ogr_add_geometry1(dst, geom, [idx, trg_index, geom.Area() / trg_area])

                # neccessary?
                dst.SyncToDisk()

        t2 = dt.datetime.now()
        #acc = acc.total_seconds()
        #acc1 = acc1.total_seconds()
        #acc2 = acc2.total_seconds()
        print "Getting Weights takes: {0} seconds".format((t2 - t1).total_seconds())#, acc+acc1, acc1+acc2, acc2-acc)


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
    # @property
    # def isec(self):
    #     """ Returns intersections
    #
    #     Returns
    #     -------
    #     array : ndarray of Nx2 point coordinate arrays
    #     """
    #     return [np.array(self._get_intersection(idx=idx)) for idx in range(self.dst.GetLayerCount())]

    # @property
    # def sources(self):
    #     """ Returns source points
    #
    #     Returns
    #     -------
    #     array : ndarray of Nx2 point coordinate arrays
    #     """
    #     lyr = self.src.ds.GetLayer()
    #     lyr.ResetReading()
    #     lyr.SetSpatialFilter(None)
    #
    #     sources = []
    #     for feature in lyr:
    #         geom = feature.GetGeometryRef()
    #         poly = ogr_to_numpy(geom)
    #         sources.append(poly)
    #
    #     return np.array(sources)

    def get_sources(self, idx):
        """ Returns source points referring to target polygon idx

        Parameters
        ----------
        idx : int, index of target polygon

        Returns
        -------
        array : ndarray of Nx2 point coordinate arrays
        """
        lyr = self.dst.ds.GetLayerByName('dst')#_{0}'.format(idx))
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter('trg={0}'.format(idx))

        index = [feature.GetField('index') for feature in lyr]

        lyr = self.src.ds.GetLayer()
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
        layer = self.src.ds.GetLayerByName('src')
        layer.ResetReading()

        # if given, we apply a buffer value to the target polygon filter
        #trg_index = trg.GetField('index')
        trg = trg.GetGeometryRef()
        trg = trg.Buffer(self._buffer)
        layer.SetSpatialFilter(trg)

        feat_cnt = layer.GetFeatureCount()

        if feat_cnt:
            ix_ = [ogr_add_geometry(dst, ogr_src.GetGeometryRef(),
                                    ogr_src.GetField('index'),  1. / feat_cnt) for ogr_src in layer]
            #ix_ = [ogr_add_geometry1(dst, ogr_src.GetGeometryRef(), [ogr_src.GetField('index'), trg_index,  1. / feat_cnt])
            #       for ogr_src in layer]
        else:
            layer.SetSpatialFilter(None)
            src_pts = np.array([ogr_src.GetGeometryRef().GetPoint_2D(0) for ogr_src in layer])
            centroid = get_centroid(trg)
            tree = cKDTree(src_pts)
            distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
            feat = layer.GetFeature(ixnext)
            ogr_add_geometry(dst, feat.GetGeometryRef(), feat.GetField('index'), 1.)
            #ogr_add_geometry1(dst, feat.GetGeometryRef(), [feat.GetField('index'), trg_index, 1.])

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

        print("JHFKJGFHGFHGFJHGF")
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
        lyr = self.zdata.src.ds.GetLayerByName('src')
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

        print(self.zdata, out)
        if self.zdata is not None:
            self.zdata.trg.set_attribute('mean', out)

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
            self.zdata.trg.set_attribute('var', out)

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


def gdal_create_dataset(drv, name, cols, rows, type, remove=False):
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

    drv = gdal.GetDriverByName('GTiff')
    if remove:
        if os.path.exists(name):
            drv.Delete(name)
    ds = drv.Create(name, cols, rows, 1, type)#gdal.GDT_Float32)


    #drv = ogr.GetDriverByName(drv)
    #if remove:
    #    if os.path.exists(name):
    #        drv.DeleteDataSource(name)
    #ds = drv.CreateDataSource(name)

    return ds


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
    if fields is not None:
        print( fields)
        for fname, fvalue in fields:
            print(fname)
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
    try:
        src_lyr = src_ds.GetLayerByName(name)
    except:
        src_lyr = src_ds.GetLayerByIndex(name)
    if reset:
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        src_lyr.SetAttributeFilter(None)
    tmp_lyr = dst_ds.CopyLayer(src_lyr, src_lyr.GetName())


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


def ogr_add_geometry1(layer, geom, attrs):
    """ Copes single OGR.Geometry object to an OGR.Layer object.

    .. versionadded:: 0.7.0

    Given OGR.Geometry is copied to new OGR.Feature and
    written to given OGR.Layer by given index. Attributes are attached.

    Parameters
    ----------
    layer : OGR.Layer object

    geom : OGR.Geometry object

    attrs : list, attributes referring to layer fields

    """
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)

    for i, item in enumerate(attrs):
        feat.SetField(i, item)
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


def numpy_to_pathpatch(arr):
    """ Returns intersections

    Returns
    -------
    array : ndarray of matplotlib.patches.PathPatch objects
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
            path = Path(item, [1] + (len(item)-1) * [2])
            paths.append(patches.PathPatch(path))

    return np.array(paths)


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
