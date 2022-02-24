#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
GDAL Raster/Vector Data I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    "VectorSource",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import os
import tempfile
import warnings

import numpy as np

from wradlib import georef
from wradlib.util import import_optional

osr = import_optional("osgeo.osr")
ogr = import_optional("osgeo.ogr")
gdal = import_optional("osgeo.gdal")

# check windows
isWindows = os.name == "nt"


def open_vector(filename, driver=None, layer=0):
    """Open vector file, return gdal.Dataset and OGR.Layer

        .. warning:: dataset and layer have to live in the same context,
            if dataset is deleted all layer references will get lost

    Parameters
    ----------
    filename : str
        vector file name
    driver : str
        gdal driver string
    layer : int or str

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.Dataset
    layer : :py:class:`gdal:osgeo.ogr.Layer`
        ogr.Layer
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
    filename : str
        raster file name
    driver : str
        gdal driver string

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.Dataset
    """

    dataset = gdal.OpenEx(filename)

    if driver:
        gdal.GetDriverByName(driver)

    return dataset


def read_safnwc(filename):
    """Read MSG SAFNWC hdf5 file into a gdal georeferenced object

    Parameters
    ----------
    filename : str
        satellite file name

    Returns
    -------
    ds : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.DataSet with satellite data
    """

    root = gdal.Open(filename)
    ds1 = gdal.Open("HDF5:" + filename + "://CT")
    ds = gdal.GetDriverByName("MEM").CreateCopy("out", ds1, 0)

    try:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(ds.GetMetadata()["PROJECTION"])
    except KeyError:
        raise KeyError("WRADLIB: Projection is missing for satellite file {filename}")

    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj.ExportToWkt())
    ds.SetGeoTransform([float(x) for x in geotransform])

    return ds


def gdal_create_dataset(
    drv, name, cols=0, rows=0, bands=0, gdal_type=None, remove=False
):
    """Creates GDAL.DataSet object.

    Parameters
    ----------
    drv : str
        GDAL driver string
    name : str
        path to filename
    cols : int
        # of columns
    rows : int
        # of rows
    bands : int
        # of raster bands
    gdal_type : :py:class:`gdal:osgeo.ogr.DataType`
        raster data type  eg. gdal.GDT_Float32
    remove : bool
        if True, existing gdal.Dataset will be
        removed before creation

    Returns
    -------
    out : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.Dataset
    """
    if gdal_type is None:
        gdal_type = gdal.GDT_Unknown

    driver = gdal.GetDriverByName(drv)
    metadata = driver.GetMetadata()

    if not metadata.get("DCAP_CREATE", False):
        raise TypeError(f"WRADLIB: Driver {drv} doesn't support Create() method.")

    if remove:
        if os.path.exists(name):
            driver.Delete(name)
    ds = driver.Create(name, cols, rows, bands, gdal_type)

    return ds


def write_raster_dataset(fpath, dataset, rformat, options=None, remove=False):
    """Write raster dataset to file format

    Parameters
    ----------
    fpath : str
        A file path - should have file extension corresponding to format.
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.Dataset  gdal raster dataset
    rformat : str
        gdal raster format string
    options : list
        List of option strings for the corresponding format.
    remove : bool
        if True, existing gdal.Dataset will be
        removed before creation

    Note
    ----
    For format and options refer to
    `formats_list <https://gdal.org/formats_list.html>`_.

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
            f"WRADLIB: Raster Driver {rformat} doesn't support CreateCopy() method."
        )

    if remove:
        if os.path.exists(fpath):
            driver.Delete(fpath)

    target = driver.CreateCopy(fpath, dataset, 0, options)
    del target


class VectorSource:
    """ DataSource class for handling ogr/gdal vector data

    DataSource handles creates in-memory (vector) ogr DataSource object with
    one layer for point or polygon geometries.

    Parameters
    ----------
    data : sequence or str
        sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        Vector File (GDAL/OGR)  filename containing source points/polygons
    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        SRS describing projection source data should be projected to

    Keyword Arguments
    -----------------
    name : str
        Layer Name, defaults to "layer".
    source : int
        Number of layer to load, if multiple layers in source shape file.
    mode : str
        Return type of class access functions/properties.
        Can be either of "numpy", "geo" and "ogr", defaults to "numpy".
    projection_source : :py:class:`gdal:osgeo.osr.SpatialReference`
        SRS describing projection source in which data is provided in.

    Warning
    -------
    Writing shapefiles with the wrong locale settings can have impact on the
    type of the decimal. If problem arise use ``LC_NUMERIC=C`` in your environment.

    Examples
    --------
    See \
    :ref:`/notebooks/fileio/wradlib_vector_source.ipynb`.
    """

    def __init__(self, data=None, srs=None, name="layer", source=0, **kwargs):
        self._srs = srs
        self._name = name
        self._geo = None
        self._mode = kwargs.get("mode", "numpy")
        self._src_srs = kwargs.get("projection_source", None)
        if data is not None:
            if isinstance(data, (np.ndarray, list)):
                self._ds = self._check_src(data)
            else:
                self.load_vector(data, source=source)
            self._create_spatial_index()
        else:
            self._ds = None

    def close(self):
        if self._geo is not None:
            self._geo = None
        if self.ds is not None:
            self.ds = None

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        """Return Layer Feature Iterator."""
        if self._mode == "ogr":
            lyr = self.ds.GetLayer()
            return iter(lyr)
        elif self._mode == "geo":
            return self.geo.iterrows()
        else:
            lyr = self.ds.GetLayer()

            def _get_geom(feat):
                return georef.ogr_to_numpy(feat.GetGeometryRef())

            return iter(map(_get_geom, lyr))

    def __len__(self):
        lyr = self.ds.GetLayer()
        return lyr.GetFeatureCount()

    def __repr__(self):
        lyr = self.ds.GetLayer()
        summary = [f"<wradlib.{type(self).__name__}>"]
        geom_type = f"Type: {ogr.GeometryTypeToName(lyr.GetGeomType())}"
        summary.append(geom_type)
        geoms = f"Geometries: {len(self)}"
        summary.append(geoms)
        return "\n".join(summary)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def ds(self):
        """Returns VectorSource"""
        self._check_ds()
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds = value

    def _check_ds(self):
        """Raise ValueError if empty VectorSource"""
        if self._ds is None:
            raise ValueError("Trying to access empty VectorSource.")

    @property
    def extent(self):
        return self.ds.GetLayer().GetExtent()

    @property
    def crs(self):
        return self.ds.GetLayer().GetSpatialRef()

    @property
    def data(self):
        """Returns VectorSource geometries as numpy arrays

        Note
        ----
        This may be slow, because it extracts all source polygons
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        return self._get_data()

    @property
    def geo(self):
        """Returns VectorSource geometries as GeoPandas Dataframe"""
        self._check_ds()
        if self._geo is None:
            geopandas = import_optional("geopandas")
            self._geo = geopandas.read_file(self.ds.GetDescription())
        return self._geo

    def _get_data(self, mode=None):
        """Returns DataSource geometries

        Keyword Arguments
        -----------------
        mode : str
            return type ("numpy", "geo", "ogr"), defaults to "numpy"
        """
        if mode is None:
            mode = self._mode
        lyr = self.ds.GetLayer()
        sources = []
        for feature in lyr:
            geom = feature.GetGeometryRef()
            if mode == "numpy":
                poly = georef.vector.ogr_to_numpy(geom)
                sources.append(poly)
            else:
                poly = geom
                sources.append(poly)
        return np.array(sources, dtype=object)

    def get_data_by_idx(self, idx, mode=None):
        """Returns DataSource geometries from given index

        Parameters
        ----------
        idx : sequence
            sequence of int indices

        Keyword Arguments
        -----------------
        mode : str
            return type ("numpy", "geo", "ogr"), defaults to "numpy"
        """
        if mode is None:
            mode = self._mode
        if mode == "geo":
            if isinstance(idx, (list, slice)):
                return self.geo.loc[idx]
            elif np.isscalar(idx):
                return self.geo.iloc[idx]
            else:
                return self.geo.loc[idx]
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        sources = []
        for i in idx:
            feature = lyr.GetFeature(i)
            geom = feature.GetGeometryRef()
            poly = georef.vector.ogr_to_numpy(geom)
            # need to recreate the geometry because access is lost if layer gets out of scope
            if mode == "ogr":
                poly = georef.vector.numpy_to_ogr(
                    poly, geom.GetGeometryName().capitalize()
                )
            sources.append(poly)
        return np.array(sources, dtype=object)

    def get_data_by_att(self, attr=None, value=None, mode=None):
        """Returns DataSource geometries filtered by given attribute/value

        Keyword Arguments
        -----------------
        attr : str
            attribute name
        value : str
            attribute value
        mode : str
            return type ("numpy", "geo", "ogr"), defaults to "numpy"
        """
        if mode is None:
            mode = self._mode
        if np.isscalar(value):
            sql = f"{attr}={value}"
        else:
            sql = f"{attr} in {tuple(value)}"
        if mode == "geo":
            return self.geo.query(sql)

        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(sql)
        return self._get_data(mode=mode)

    def get_data_by_geom(self, geom=None, mode=None):
        """Returns DataSource geometries filtered by given geometry

        Keyword Arguments
        -----------------
        geom : :py:class:`gdal:osgeo.ogr.Geometry` | :py:class:`geopandas.GeoDataFrame`
            OGR.Geometry object or geopandas.GeoDataFrame containing the Geometry
        mode : str
            return type ("numpy", "geo", "ogr"), defaults to "numpy"
        """
        if mode is None:
            mode = self._mode
        if mode == "geo":
            return self.geo[self.geo.within(geom)]
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetAttributeFilter(None)
        lyr.SetSpatialFilter(geom)
        return self._get_data(mode=mode)

    def _create_spatial_index(self):
        """Creates spatial index file .qix"""
        sql1 = f"DROP SPATIAL INDEX ON {self._name}"
        sql2 = f"CREATE SPATIAL INDEX ON {self._name}"
        self.ds.ExecuteSQL(sql1)
        self.ds.ExecuteSQL(sql2)

    def _create_table_index(self, col):
        """Creates attribute index files"""
        sql1 = f"DROP INDEX ON {self._name}"
        sql2 = f"CREATE INDEX ON {self._name} USING {col}"
        self.ds.ExecuteSQL(sql1)
        self.ds.ExecuteSQL(sql2)

    def _check_src(self, src):
        """Basic check of source elements (sequence of points or polygons).

        - array cast of source elements
        - create ogr_src datasource/layer holding src points/polygons
        - transforming source grid points/polygons to ogr.geometries
          on ogr.layer
        """
        tmpfile = tempfile.NamedTemporaryFile(mode="w+b").name
        ogr_src = gdal_create_dataset(
            "ESRI Shapefile", os.path.join("/vsimem", tmpfile), gdal_type=gdal.OF_VECTOR
        )
        src = np.array(src)
        if self._src_srs is not None:
            src = georef.reproject(
                src, projection_source=self._src_srs, projection_target=self._srs
            )
        # create memory datasource, layer and create features
        if src.ndim == 2:
            geom_type = ogr.wkbPoint
        # no Polygons, just Points
        else:
            geom_type = ogr.wkbPolygon
        fields = [("index", ogr.OFTInteger)]
        georef.vector.ogr_create_layer(
            ogr_src, self._name, srs=self._srs, geom_type=geom_type, fields=fields
        )
        georef.vector.ogr_add_feature(ogr_src, src, name=self._name)

        return ogr_src

    def dump_vector(self, filename, driver="ESRI Shapefile", remove=True):
        """Output layer to OGR Vector File

        Parameters
        ----------
        filename : str
            path to shape-filename

        Keyword Arguments
        -----------------
        driver : str
            driver string
        remove : bool
            if True removes existing output file
        """
        ds_out = gdal_create_dataset(
            driver, filename, gdal_type=gdal.OF_VECTOR, remove=remove
        )
        georef.vector.ogr_copy_layer(self.ds, 0, ds_out)

        # flush everything
        del ds_out

    def load_vector(self, filename, source=0, driver="ESRI Shapefile"):
        """Read Layer from OGR Vector File

        Parameters
        ----------
        filename : str
            path to shape-filename

        Keyword Arguments
        -----------------
        source : int or str
            number or name of wanted layer, defaults to 0
        driver : str
            driver string
        """
        tmpfile = tempfile.NamedTemporaryFile(mode="w+b").name
        self.ds = gdal_create_dataset(
            "ESRI Shapefile", os.path.join("/vsimem", tmpfile), gdal_type=gdal.OF_VECTOR
        )
        # get input file handles
        ds_in, tmp_lyr = open_vector(filename, driver=driver, layer=source)

        # get spatial reference object
        srs = tmp_lyr.GetSpatialRef()
        # fall back to given projection
        if srs is None:
            srs = self._src_srs

        # raise error as we can't do anything about it
        if self._srs is None and srs is None:
            raise ValueError(
                f"Spatial reference missing from source file {filename}. "
                f"Please provide a fitting spatial reference object"
            )

        # this will be combined with the above the future to raise unconditionally
        if srs is None:
            warnings.warn(
                f"Spatial reference missing from source file {filename}. "
                f"This will raise an error from wradlib version 2.0",
                FutureWarning,
            )

        # reproject layer if necessary
        if self._srs is not None and srs is not None and srs != self._srs:
            ogr_src_lyr = self.ds.CreateLayer(
                self._name, self._srs, geom_type=ogr.wkbPolygon
            )
            georef.vector.ogr_reproject_layer(
                tmp_lyr, ogr_src_lyr, self._srs, src_srs=srs
            )
        else:
            # copy layer
            ogr_src_lyr = self.ds.CopyLayer(tmp_lyr, self._name)
            if self._srs is None:
                self._srs = srs

        # flush everything
        del ds_in

    def dump_raster(
        self, filename, driver="GTiff", attr=None, pixel_size=1.0, remove=True, **kwargs
    ):
        """Output layer to GDAL Rasterfile

        Parameters
        ----------
        filename : str
            path to shape-filename

        Keyword Arguments
        -----------------
        driver : str
            GDAL Raster Driver
        attr : str
            attribute to burn into raster
        pixel_size : float
            pixel Size in source units
        remove : bool
            if True removes existing output file
        silent : bool
            If True no ProgressBar is shown. Defaults to False.
        """
        silent = kwargs.pop("silent", False)
        progress = None if (silent or isWindows) else gdal.TermProgress

        layer = self.ds.GetLayer()
        layer.ResetReading()

        x_min, x_max, y_min, y_max = layer.GetExtent()

        cols = int((x_max - x_min) / pixel_size)
        rows = int((y_max - y_min) / pixel_size)

        # Todo: at the moment, always writing floats
        ds_out = gdal_create_dataset(
            "MEM", "", cols, rows, 1, gdal_type=gdal.GDT_Float32
        )

        ds_out.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        proj = layer.GetSpatialRef()
        if proj is None:
            proj = self._srs
        ds_out.SetProjection(proj.ExportToWkt())

        band = ds_out.GetRasterBand(1)
        band.FlushCache()
        if attr is not None:
            gdal.RasterizeLayer(
                ds_out,
                [1],
                layer,
                burn_values=[0],
                options=[f"ATTRIBUTE={attr}", "ALL_TOUCHED=TRUE"],
                callback=progress,
            )
        else:
            gdal.RasterizeLayer(
                ds_out,
                [1],
                layer,
                burn_values=[1],
                options=["ALL_TOUCHED=TRUE"],
                callback=progress,
            )

        write_raster_dataset(filename, ds_out, driver, remove=remove)

        del ds_out

    def set_attribute(self, name, values, reset_filter=False):
        """Add/Set given Attribute with given values

        Parameters
        ----------
        name : str
            Attribute Name
        values : :class:`numpy:numpy.ndarray`
            Values to fill in attributes

        Keyword Arguments
        -----------------
        reset_filter : bool
            reset any layer filter (spatial/attribute), defaults to False
        """
        lyr = self.ds.GetLayerByIndex(0)
        if reset_filter:
            lyr.SetAttributeFilter(None)
            lyr.SetSpatialFilter(None)
        lyr.ResetReading()
        # todo: automatically check for value type
        defn = lyr.GetLayerDefn()

        if defn.GetFieldIndex(name) == -1:
            lyr.CreateField(ogr.FieldDefn(name, ogr.OFTReal))

        for i, item in enumerate(lyr):
            item.SetField(name, values[i])
            lyr.SetFeature(item)

        lyr.SyncToDisk()
        self._geo = None

    def get_attributes(self, attrs, filt=None):
        """Return attributes

        Parameters
        ----------
        attrs : list
            Attribute Names to retrieve

        Keyword Arguments
        -----------------
        filt : tuple
            (attname, value) for Attribute Filter
        """

        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetAttributeFilter(None)
        lyr.SetSpatialFilter(None)
        if filt is not None:
            lyr.SetAttributeFilter(f"{filt[0]}={filt[1]}")
        ret = [[] for _ in attrs]
        for ogr_src in lyr:
            for i, att in enumerate(attrs):
                ret[i].append(ogr_src.GetField(att))
        return ret

    def get_geom_properties(self, props, filt=None):
        """Return geometry properties

        Parameters
        ----------
        props : list
            Property Names to retrieve

        Keyword Arguments
        -----------------
        filt : tuple
            (attname, value) for Attribute Filter
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        if filt is not None:
            lyr.SetAttributeFilter(f"{filt[0]}={filt[1]}")
        ret = [[] for _ in props]
        for ogr_src in lyr:
            for i, prop in enumerate(props):
                ret[i].append(getattr(ogr_src.GetGeometryRef(), prop)())
        return ret

    def get_attrs_and_props(self, attrs=None, props=None, filt=None):
        """Return properties and attributes

        Keyword Arguments
        -----------------
        attrs : list
           Attribute Names to retrieve
        props : list
           Property Names to retrieve
        filt : tuple
           (attname, value) for Attribute Filter
        """
        lyr = self.ds.GetLayer()
        lyr.ResetReading()
        if filt is not None:
            lyr.SetAttributeFilter(f"{filt[0]}={filt[1]}")
        ret_props = [[] for _ in props]
        ret_attrs = [[] for _ in attrs]
        for ogr_src in lyr:
            for i, att in enumerate(attrs):
                ret_attrs[i].append(ogr_src.GetField(att))
            for i, prop in enumerate(props):
                ret_props[i].append(getattr(ogr_src.GetGeometryRef(), prop)())

        return ret_attrs, ret_props
