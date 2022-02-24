#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Zonal Statistics
^^^^^^^^^^^^^^^^

This module supports you in computing statistics over spatial zones. A typical
application would be to compute mean areal precipitation for a catchment by
using precipitation estimates from a radar grid in polar coordinates or from
precipitation estimates in a Cartesian grid.

The general usage is similar to the :mod:`wradlib.ipol` and
:mod:`wradlib.adjust`:

You have to create an instance of a class (derived from
:class:`~wradlib.zonalstats.ZonalDataBase`) by using
the spatial information of your source and target objects (e.g. radar bins and
catchment polygons). The Zonal Data within this object can be saved eg. as an
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

   {}
"""
__all__ = [
    # "DataSource",
    "ZonalDataBase",
    "ZonalDataPoint",
    "ZonalDataPoly",
    "ZonalStatsBase",
    "ZonalStatsPoly",
    "ZonalStatsPoint",
    "mask_from_bbox",
    "get_bbox",
    "grid_centers_to_vertices",
    "get_clip_mask",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import os
import tempfile
import warnings

import numpy as np
from scipy import spatial

from wradlib import georef, io
from wradlib.util import has_import, import_optional

ogr = import_optional("osgeo.ogr")
osr = import_optional("osgeo.osr")
gdal = import_optional("osgeo.gdal")
mpl_patches = import_optional("matplotlib.patches")
mpl_path = import_optional("matplotlib.path")

if has_import(gdal):
    ogr.UseExceptions()
    gdal.UseExceptions()

# check windows
isWindows = os.name == "nt"


class DataSource(io.VectorSource):
    """DataSource class for handling ogr/gdal vector data

    Minimal wrapper around wradlib.io.VectorSource for backwards compatibility.
    """


class ZonalDataBase:
    """Base class for managing 2-dimensional zonal data.

    For target polygons from either source points or source polygons.
    Provides the basic design for all other classes.

    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the created destination layer will be empty.

    Data Model is built upon OGR Implementation of ESRI Shapefile

    * one src DataSource (named 'src') holding source polygons or points
    * one trg DataSource (named 'trg') holding target polygons
    * one dst DataSource (named 'dst') holding intersection polygons/points
      related to target polygons with attached index and weights fields

    By using OGR there are no restrictions for the used source grids.

    Warning
    -------
    Writing shapefiles with the wrong locale settings can have impact on the
    type of the decimal. If problem arise use LC_NUMERIC=C in your environment.

    Parameters
    ----------
    src : sequence or str
        sequence of source points (shape Nx2) or polygons (shape NxMx2) or
        ESRI Shapefile filename containing source points/polygons or
        DataSource object
    trg : sequence or str
        sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons or DataSource object

    Keyword arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Points/Polygons  will be considered inside the target if they are
        contained in the buffer.

    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        OGR.SpatialReference will be used for DataSource object.
        src and trg data have to be in the same srs-format

    silent : bool
        If True no ProgressBar is shown. Defaults to False.


    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.

    """

    def __init__(self, src, trg=None, buf=0.0, srs=None, **kwargs):
        self._buffer = buf
        self._srs = srs
        silent = kwargs.pop("silent", False)

        if trg is None:
            # try to read complete dump (src, trg, dst)
            self.load_vector(src)
        else:
            if isinstance(src, io.VectorSource):
                self.src = src
            else:
                self.src = io.VectorSource(src, name="src", srs=srs, **kwargs)

            if isinstance(trg, io.VectorSource):
                self.trg = trg
            else:
                self.trg = io.VectorSource(trg, name="trg", srs=srs, **kwargs)

            self.dst = io.VectorSource(name="dst")
            self.dst.ds = self._create_dst_datasource(silent)
            self.dst._create_spatial_index()

        self.dst._create_table_index("trg_index")
        self._count_intersections = self.dst.ds.GetLayer().GetFeatureCount()

    @property
    def count_intersections(self):
        """Returns number of intersections"""
        return self._count_intersections

    @property
    def srs(self):
        """Returns SpatialReferenceSystem object"""
        return self._srs

    @property
    def isecs(self):
        """Returns intersections

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            Array of Nx2 point coordinate arrays
        """
        return np.array(
            [
                self._get_intersection(idx=idx)
                for idx in range(self.trg.ds.GetLayerByName("trg").GetFeatureCount())
            ],
            dtype=object,
        )

    def get_isec(self, idx):
        """Returns intersections

        Parameters
        ----------
        idx : int
            index of target polygon

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            Array of Nx2 point coordinate arrays
        """
        return self._get_intersection(idx=idx)

    def get_source_index(self, idx):
        """Returns source indices referring to target polygon idx

        Parameters
        ----------
        idx : int
            index of target polygon

        Returns
        -------
        array : :class:`numpy:numpy.ndarray`
            indices
        """
        return np.array(
            self.dst.get_attributes(["src_index"], filt=("trg_index", idx))[0]
        )

    def _create_dst_datasource(self, silent):
        """Create destination target gdal.Dataset

        Creates one layer for each target polygon, consisting of
        the needed source data attributed with index and weights fields

        Returns
        -------
        ds_mem : :py:class:`gdal:osgeo.gdal.Dataset`
            gdal.Dataset object
        """
        progress = None if (silent or isWindows) else gdal.TermProgress

        # create mem-mapped temp file dataset
        tmpfile = tempfile.NamedTemporaryFile(mode="w+b").name
        ds_out = io.gdal.gdal_create_dataset(
            "ESRI Shapefile", os.path.join("/vsimem", tmpfile), gdal_type=gdal.OF_VECTOR
        )

        # create intermediate mem dataset
        ds_mem = io.gdal.gdal_create_dataset("Memory", "out", gdal_type=gdal.OF_VECTOR)

        # get src geometry layer
        src_lyr = self.src.ds.GetLayerByName("src")
        src_lyr.ResetReading()
        src_lyr.SetSpatialFilter(None)
        geom_type = src_lyr.GetGeomType()

        # get trg geometry layer
        trg_lyr = self.trg.ds.GetLayerByName("trg")
        trg_lyr.ResetReading()
        trg_lyr.SetSpatialFilter(None)

        # buffer handling (time consuming)
        if self._buffer > 0:
            for i in range(trg_lyr.GetFeatureCount()):
                feat = trg_lyr.GetFeature(i)
                feat.SetGeometryDirectly(feat.GetGeometryRef().Buffer(self._buffer))
                trg_lyr.SetFeature(feat)

        # reset target layer
        trg_lyr.ResetReading()

        # create tmp dest layer
        self.tmp_lyr = georef.vector.ogr_create_layer(
            ds_mem, "dst", srs=self._srs, geom_type=geom_type
        )

        trg_lyr.Intersection(
            src_lyr,
            self.tmp_lyr,
            options=[
                "SKIP_FAILURES=YES",
                "INPUT_PREFIX=trg_",
                "METHOD_PREFIX=src_",
                "PROMOTE_TO_MULTI=YES",
                "USE_PREPARED_GEOMETRIES=YES",
                "PRETEST_CONTAINMENT=YES",
            ],
            callback=progress,
        )

        georef.vector.ogr_copy_layer(ds_mem, 0, ds_out)

        return ds_out

    def dump_vector(self, filename, driver="ESRI Shapefile", remove=True):
        """Output source/target grid points/polygons to ESRI_Shapefile

        target layer features are attributed with source index and weight

        Parameters
        ----------
        filename : str
            path to shape-filename
        driver : str
            OGR Vector Driver String, defaults to 'ESRI Shapefile'
        remove : bool
            if True, existing file will be removed before creation
        """
        self.src.dump_vector(filename, driver, remove=remove)
        self.trg.dump_vector(filename, driver, remove=False)
        self.dst.dump_vector(filename, driver, remove=False)

    def load_vector(self, filename):
        """Load source/target grid points/polygons into in-memory Shapefile

        Parameters
        ----------
        filename : str
            path to vector file
        """
        self.src = io.VectorSource(filename, name="src", source="src")
        self.trg = io.VectorSource(filename, name="trg", source="trg")
        self.dst = io.VectorSource(filename, name="dst", source="dst")

        # get spatial reference object
        self._srs = self.src.ds.GetLayer().GetSpatialRef()

    def _get_idx_weights(self):
        """Retrieve index and weight from dst DataSource"""
        raise NotImplementedError

    def _get_intersection(self, trg=None, idx=None, buf=0.0):
        """Just a toy function if you want to inspect the intersection
        points/polygons of an arbitrary target or an target by index.
        """
        # TODO: kwargs necessary?

        # check wether idx is given
        if idx is not None:
            if self.trg:
                try:
                    lyr = self.trg.ds.GetLayerByName("trg")
                    feat = lyr.GetFeature(idx)
                    trg = feat.GetGeometryRef()
                except Exception:
                    raise TypeError(f"No target polygon found at index {idx}")
            else:
                raise TypeError("No target polygons found in object!")

        # check for trg
        if trg is None:
            raise TypeError("Either *trg* or *idx* keywords must be given!")

        # check for geometry
        if not type(trg) == ogr.Geometry:
            trg = georef.vector.numpy_to_ogr(trg, "Polygon")

        # apply Buffer value
        trg = trg.Buffer(buf)

        if idx is None:
            intersecs = self.dst.get_data_by_geom(trg)
        else:
            intersecs = self.dst.get_data_by_att("trg_index", idx)

        return intersecs


class ZonalDataPoly(ZonalDataBase):
    """ZonalData object for source polygons

    Parameters
    ----------
    src : sequence or str
        sequence of source polygons (shape NxMx2) or
        ESRI Shapefile filename containing source polygons

    trg : sequence or str
        sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword Arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Polygons will be considered inside the target if they are contained
        in the buffer.

    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        OGR.SpatialReference will be used for DataSource object.
        src and trg data have to be in the same srs-format

    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.
    """

    def _get_idx_weights(self):
        """Retrieve index and weight from dst DataSource

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
            arr, w = self.dst.get_attrs_and_props(
                attrs=["src_index"], props=["Area"], filt=("trg_index", index)
            )
            arr.append(w[0])
            for i, l in enumerate(arr):
                ret[i].append(np.array(l))
        return tuple(ret)


class ZonalDataPoint(ZonalDataBase):
    """ZonalData object for source points

    Parameters
    ----------
    src : sequence or str
        sequence of source points (shape Nx2) or
        ESRI Shapefile filename containing source points
    trg : sequence or str
        sequence of target polygons (shape Nx2, num vertices x 2) or
        ESRI Shapefile filename containing target polygons

    Keyword Arguments
    -----------------
    buf : float
        (same unit as coordinates)
        Points will be considered inside the target if they are contained
        in the buffer.

    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        OGR.SpatialReference will be used for DataSource object.
        src and trg data have to be in the same srs-format

    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalData`.
    """

    def _get_idx_weights(self):
        """Retrieve index and weight from dst DataSource

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
            arr = self.dst.get_attributes(["src_index"], filt=("trg_index", index))
            arr.append([1.0 / len(arr[0])] * len(arr[0]))
            for i, l in enumerate(arr):
                ret[i].append(np.array(l))
        return tuple(ret)


class ZonalStatsBase:
    """Base class for all 2-dimensional zonal statistics.

    The base class for computing 2-dimensional zonal statistics for target
    polygons from source points or polygons as built up with ZonalDataBase
    and derived classes. Provides the basic design for all other classes.

    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the zonal statistic for that target will be NaN.

    Parameters
    ----------
    src : :class:`wradlib.zonalstats.ZonalDataPoly` or str
        ZonalDataPoly object or filename pointing to ZonalDataPoly ESRI
        shapefile containing necessary ZonalData
        ZonalData is available as ``zdata``-property inside class instance.

    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalStats`.
    """

    def __init__(self, src=None, ix=None, w=None):

        self._ix = None
        self._w = None

        if src is not None:
            if isinstance(src, ZonalDataBase):
                if not src.count_intersections:
                    raise ValueError(
                        "No intersections found in destination "
                        "layer of ZonalDataBase object."
                    )
                self._zdata = src
            else:
                raise TypeError("Parameter mismatch in calling ZonalDataBase")
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
        """ """
        isempty = np.repeat(False, len(self.w))
        for i, weights in enumerate(self.w):
            if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
                isempty[i] = True
        return isempty

    def _check_ix_w(self, ix, w):
        """TODO Basic check of target attributes (sequence of values)."""
        if ix is not None and w is not None:
            if len(ix) != len(w):
                raise TypeError("parameters ix and w must be of equal length")
            return np.array(ix, dtype=object), np.array(w, dtype=object)

        else:
            raise TypeError(
                "ix and w are complementary parameters and " "must both be given"
            )

    def _check_vals(self, vals):
        """TODO Basic check of target elements (sequence of polygons)."""
        if self.zdata is not None:
            lyr = self.zdata.src.ds.GetLayerByName("src")
            lyr.ResetReading()
            lyr.SetSpatialFilter(None)
            src_len = lyr.GetFeatureCount()
            assert len(vals) == src_len, f"Argument vals must be of length {src_len}"
        else:
            imax = 0
            for i in self.ix:
                mx = np.nanmax(i)
                if imax < mx:
                    imax = mx
            assert (
                len(vals) > imax
            ), "Argument vals cannot be subscripted by given index values"

        return vals

    def mean(self, vals):
        """Evaluate (weighted) zonal mean for values given at the source \
        points.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            1-d array of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        self.isempty = self.check_empty()
        out = np.zeros(len(self.ix)) * np.nan
        out[~self.isempty] = np.array(
            [
                np.average(vals[self.ix[i].astype(int)], weights=self.w[i])
                for i in np.arange(len(self.ix))[~self.isempty]
            ]
        )
        if self.zdata is not None:
            self.zdata.trg.set_attribute("mean", out)

        return out

    def var(self, vals):
        """Evaluate (weighted) zonal variance for values given at the source \
        points.

        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            1-d  array of type float with the same length as self.src
            Values at the source element for which to compute
            zonal statistics

        """
        self._check_vals(vals)
        mean = self.mean(vals)
        out = np.zeros(len(self.ix)) * np.nan
        out[~self.isempty] = np.array(
            [
                np.average(
                    (vals[self.ix[i].astype(int)] - mean[i]) ** 2, weights=self.w[i]
                )
                for i in np.arange(len(self.ix))[~self.isempty]
            ]
        )

        if self.zdata is not None:
            self.zdata.trg.set_attribute("var", out)

        return out


class ZonalStatsPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    Parameters
    ----------
    src : :class:`~wradlib.zonalstats.ZonalDataPoly` or str
        ZonalDataPoly object or filename pointing to ZonalDataPoly ESRI
        shapefile containing necessary Zonal Data

    Keyword arguments
    -----------------

    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalStats`
    and :ref:`/notebooks/zonalstats/wradlib_zonalstats_example.ipynb`.
    """

    def __init__(self, src=None, **kwargs):
        if src is not None:
            if not isinstance(src, ZonalDataPoly):
                src = ZonalDataPoly(src, **kwargs)
        super().__init__(src, **kwargs)


class ZonalStatsPoint(ZonalStatsBase):
    """Compute zonal average from all points in or close to the target polygon.

    Parameters
    ----------
    src : :class:`~wradlib.zonalstats.ZonalDataPoint` or str
        ZonalDataPoint object or filename pointing to ZonalDataPoly ESRI
        shapefile containing necessary Zonal Data

    Keyword arguments
    -----------------

    Examples
    --------
    See \
    :ref:`/notebooks/zonalstats/wradlib_zonalstats_classes.ipynb#ZonalStats`
    and :ref:`/notebooks/zonalstats/wradlib_zonalstats_example.ipynb`.
    """

    def __init__(self, src, **kwargs):
        if src is not None:
            if not isinstance(src, ZonalDataPoint):
                src = ZonalDataPoint(src, **kwargs)
        super().__init__(src, **kwargs)


def numpy_to_pathpatch(arr):
    """Returns PathPatches from nested array

    Parameters
    ----------
    arr : :class:`numpy:numpy.ndarray`
        array of Polygon/Multipolygon vertices

    Returns
    -------
    array : :class:`numpy:numpy.ndarray`
        array of matplotlib.patches.PathPatch objects
    """
    paths = []
    for item in arr:
        if item.ndim != 2:
            vert = np.vstack(item)
            code = np.full(vert.shape[0], 2, dtype=np.int)
            ind = np.cumsum([0] + [len(x) for x in item[:-1]])
            code[ind] = 1
            path = mpl_path.Path(vert, code)
            paths.append(mpl_patches.PathPatch(path))
        else:
            path = mpl_path.Path(item, [1] + (len(item) - 1) * [2])
            paths.append(mpl_patches.PathPatch(path))

    return np.array(paths)


def mask_from_bbox(x, y, bbox, polar=False):
    """Return 2-d index array based on spatial selection from a bounding box.

    Use this function to create a 2-d boolean mask from 2-d arrays of grids
    points.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        x (Cartesian) coordinates of shape (num rows, num columns)
    y : :class:`numpy:numpy.ndarray`
        y (Cartesian) coordinates of shape (num rows, num columns)
    bbox : dict
        dictionary with keys "left", "right", "bottom", "top"
        These must refer to the same Cartesian reference system as x and y
    polar : bool
        if True, x, y are aligned polar (azimuth x range)

    Returns
    -------
    mask, shape : :class:`numpy:numpy.ndarray`, tuple
              mask is a boolean array that is True if the point is inside the
              bbox, shape is the shape of the True subgrid

    """
    ny, nx = x.shape

    ix = np.arange(x.size).reshape(x.shape)

    # Find bbox corners
    #    Plant a tree
    tree = spatial.cKDTree(np.vstack((x.ravel(), y.ravel())).transpose())
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
        ax = np.array([[ill, ilr], [ill, iur], [iul, ilr], [iul, iur]], dtype=int)
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
    source_angle : float or :class:`numpy:numpy.ndarray`
        starting angle
    target_angle : float or :class:`numpy:numpy.ndarray`
        target angle
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
    return dict(left=np.min(x), right=np.max(x), bottom=np.min(y), top=np.max(y))


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
    dx : float
        grid spacing in x direction
    dy : float
        grid spacing in y direction

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        3-d array of vertices for each grid cell of shape (n grid points,5, 2)
    """
    top = y + dy / 2
    left = x - dx / 2
    right = x + dy / 2
    bottom = y - dy / 2

    verts = np.vstack(
        (
            [left.ravel(), bottom.ravel()],
            [right.ravel(), bottom.ravel()],
            [right.ravel(), top.ravel()],
            [left.ravel(), top.ravel()],
            [left.ravel(), bottom.ravel()],
        )
    ).T.reshape((-1, 5, 2))

    return verts


def get_clip_mask(coords, clippoly, srs=None):
    """Returns boolean mask of points ``coords`` inside polygon ``clippoly``

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        array of xy coords with shape [...,2]
    clippoly : :class:`numpy:numpy.ndarray`
        array of xy coords with shape (N,2) representing closed
        polygon coordinates
    srs : :py:class:`gdal:osgeo.osr.SpatialReference`
        osr.SpatialReference

    Returns
    -------
    src_mask : :class:`numpy:numpy.ndarray`
        boolean array of shape coords.shape[0:-1]

    """
    clip = [clippoly]

    zd = ZonalDataPoint(coords.reshape(-1, coords.shape[-1]), clip, srs=srs)

    # Subsetting in order to use only precipitating profiles
    src_mask = np.zeros(coords.shape[0:-1], dtype=np.bool_)

    try:
        obj = ZonalStatsPoint(zd)

        # Get source indices within polygon from zonal object
        # (0 because we have only one zone)
        pr_idx = obj.zdata.get_source_index(0)
        mask = np.unravel_index(pr_idx, coords.shape[0:-1])
        src_mask[mask] = True

    except ValueError as err:
        warnings.warn(err)

    return src_mask


if __name__ == "__main__":
    print("wradlib: Calling module <zonalstats> as main...")
