#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Projection Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "reproject",
    "create_osr",
    "projstr_to_osr",
    "epsg_to_osr",
    "wkt_to_osr",
    "get_default_projection",
    "get_earth_radius",
    "get_radar_projection",
    "get_earth_projection",
    "get_extent",
    "GeorefProjectionMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from functools import singledispatch

import numpy as np
import xradar as xd
from xarray import DataArray, Dataset, apply_ufunc

from wradlib.util import docstring, import_optional

gdal = import_optional("osgeo.gdal")
ogr = import_optional("osgeo.ogr")
osr = import_optional("osgeo.osr")
pyproj = import_optional("pyproj")

# Taken from document "Radarkomposits - Projektionen und Gitter", Version 1.01
# 5th of April 2022
_radolan_ref = dict(
    sphere=dict(
        default=dict(x_0=0.0, y_0=0.0),
        rx=dict(x_0=522962.16692185635, y_0=3759144.724265574),
        de1200=dict(x_0=542962.166921856585, y_0=3609144.7242655745),
        de4800=dict(x_0=543337.16692185646, y_0=3608769.7242655735),
    ),
    wgs84=dict(
        default=dict(x_0=0.0, y_0=0.0),
        rx=dict(x_0=523196.83521777776, y_0=3772588.861931134),
        de1200=dict(x_0=543196.83521776402, y_0=3622588.8619310018),
        de4800=dict(x_0=543571.83521776402, y_0=3622213.8619310018),
    ),
)


def create_osr(projname, **kwargs):
    """Conveniently supports the construction of osr spatial reference objects

    Currently, the following projection names (argument *projname*)
    are supported:

    **"aeqd": Azimuthal Equidistant**

    needs the following keyword arguments:

        - *lat_0* (latitude at projection center),
        - *lon_0* (longitude at projection center),
        - *x_0* (false Easting, also known as x-offset),
        - *y_0* (false Northing, also known as y-offset)

    **"dwd-radolan" : RADOLAN Composite Coordinate System**

        - no additional arguments needed.

    Polar stereographic projection used by the German Weather Service (DWD)
    for all Radar composite products. See the final report on the RADOLAN
    project :cite:`DWD2004` for details.

    Parameters
    ----------
    projname : str
        "aeqd" or "dwd-radolan"
    kwargs : dict
        depends on projname - see above!

    Returns
    -------
    output : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection

    Examples
    --------
    See :ref:`/notebooks/basics/wradlib_workflow.ipynb#\
Georeferencing-and-Projection`.
    """
    aeqd_wkt = (
        'PROJCS["unnamed",'
        'GEOGCS["WGS 84",'
        'DATUM["unknown",'
        'SPHEROID["WGS84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Azimuthal_Equidistant"],'
        'PARAMETER["latitude_of_center",{0:-f}],'
        'PARAMETER["longitude_of_center",{1:-f}],'
        'PARAMETER["false_easting",{2:-f}],'
        'PARAMETER["false_northing",{3:-f}],'
        'UNIT["Meter",1]]'
    )

    wgs84_wkt = (
        'PROJCS["Radolan Projection",'
        'GEOGCS["Radolan Coordinate System",'
        'DATUM["unknown based on WGS 84",'
        'SPHEROID["WGS 84", 6378137, 298.25722356301]],'
        'PRIMEM["Greenwich", 0,'
        'AUTHORITY["EPSG", "8901"]],'
        'UNIT["degree", 0.0174532925199433,'
        'AUTHORITY["EPSG", "9122"]]],'
    )

    sphere_wkt = (
        'PROJCS["Radolan Projection",'
        'GEOGCS["Radolan Coordinate System",'
        'DATUM["Radolan_Kugel",'
        'SPHEROID["Erdkugel", 6370040, 0]],'
        'PRIMEM["Greenwich", 0,'
        'AUTHORITY["EPSG","8901"]],'
        'UNIT["degree", 0.017453292519943295,'
        'AUTHORITY["EPSG","9122"]]],'
    )

    radolan_ellps = dict(sphere=sphere_wkt, wgs84=wgs84_wkt)
    meter = 'UNIT["metre", 1,' 'AUTHORITY["EPSG", "9001"]],'
    kmeter = 'UNIT["kilometre", 1000,' 'AUTHORITY["EPSG","9036"]],'

    polar_stereo_wkt = (
        'PROJECTION["Polar_Stereographic"],'
        'PARAMETER["latitude_of_origin", 60],'
        'PARAMETER["central_meridian", 10],'
        'PARAMETER["false_easting", {0:-.16f}],'
        'PARAMETER["false_northing", {1:-.16f}],'
        "{2}"
        'AXIS["Easting", SOUTH],'
        'AXIS["Northing", SOUTH]]'
    )

    crs = osr.SpatialReference()

    if projname == "aeqd":
        # Azimuthal Equidistant
        x_0 = kwargs.get("x_0", 0.0)
        y_0 = kwargs.get("y_0", 0.0)
        if "x_0" in kwargs:
            crs.ImportFromWkt(
                aeqd_wkt.format(kwargs["lat_0"], kwargs["lon_0"], x_0, y_0)
            )
        else:
            crs.ImportFromWkt(
                aeqd_wkt.format(kwargs["lat_0"], kwargs["lon_0"], 0.0, 0.0)
            )
    elif "dwd-radolan" in projname:
        projname = projname.split("-")
        if len(projname) > 2:
            ellps = projname[2]
            unit = meter
        else:
            ellps = "sphere"
            unit = kmeter
        if len(projname) > 3:
            grid = projname[3]
        else:
            grid = "default"
        ref = _radolan_ref[ellps][grid]
        # override false easting/northing
        x_0 = kwargs.get("x_0", ref["x_0"])
        y_0 = kwargs.get("y_0", ref["y_0"])
        radolan_wkt = radolan_ellps[ellps] + polar_stereo_wkt.format(x_0, y_0, unit)
        crs.ImportFromWkt(radolan_wkt)
    else:
        raise ValueError(
            f"No convenience support for projection {projname!r} yet. "
            f"Please create projection by using other means."
        )

    return crs


def projstr_to_osr(projstr):
    """Transform a PROJ string to an osr spatial reference object

    Parameters
    ----------
    projstr : str
        PROJ string describing projection

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL OSR SRS object defining projection

    Examples
    --------

    See :ref:`/notebooks/fileio/radolan/radolan_grid.ipynb#PROJ`.

    """
    crs = osr.SpatialReference()
    crs.ImportFromProj4(projstr)
    try:
        crs.AutoIdentifyEPSG()
    except RuntimeError:
        pass

    if crs.Validate() == ogr.OGRERR_CORRUPT_DATA:
        raise ValueError(
            "`projstr` validates to 'ogr.OGRERR_CORRUPT_DATA'"
            "and can't be imported as OSR object."
        )
    return crs


@singledispatch
def reproject(*args, **kwargs):
    """Transform coordinates from a source projection to a target projection.

    Call signatures::

        reproject(C, **kwargs)
        reproject(X, Y, **kwargs)
        reproject(X, Y, Z, **kwargs)

    *C* is the np array of source coordinates.
    *X*, *Y* and *Z* specify arrays of x, y, and z coordinate values

    Parameters
    ----------
    C : :class:`numpy:numpy.ndarray`
        Array of shape (...,2) or (...,3) with coordinates (x,y) or (x,y,z)
        respectively
    X : :class:`numpy:numpy.ndarray`
        Array of x coordinates
    Y : :class:`numpy:numpy.ndarray`
        Array of y coordinates
    Z : :class:`numpy:numpy.ndarray`
        Array of z coordinates

    Keyword Arguments
    -----------------
    src_crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        defaults to EPSG(4326)
    trg_crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        defaults to EPSG(4326)
    area_of_interest : tuple
        floats (WestLongitudeDeg, SouthLatitudeDeg, EastLongitudeDeg,
        NorthLatitudeDeg), only gdal>=3, defaults to None (no area of interest)

    Returns
    -------
    trans : :class:`numpy:numpy.ndarray`
        Array of reprojected coordinates x,y (...,2) or x,y,z (...,3)
        depending on input array.
    X, Y : :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y coordinates, shape depending on input array
    X, Y, Z: :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y,z coordinates, shape depending on input array

    Examples
    --------

    See :ref:`/notebooks/georeferencing/georef.ipynb`.

    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape = C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1, numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError("Input Array column mismatch to `reproject`.")
    else:
        if len(args) == 2:
            X, Y = (np.asanyarray(arg) for arg in args)
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            numCols = 3
        else:
            raise TypeError(
                f"Illegal number arguments ({len(args)}) to "
                f"`reproject`. Should be 3 or less."
            )

        xshape = X.shape
        yshape = Y.shape

        if xshape != yshape:
            raise TypeError(
                f"Shape mismatch `X` ({xshape}), `Y` ({yshape}) inputs to `reproject`."
            )

        if "Z" in locals():
            if xshape != zshape:
                raise TypeError(
                    f"Shape mismatch `Z` ({zshape}) input to `reproject`. "
                    f"Shape of ({xshape}) needed."
                )
            C = np.concatenate(
                [X.ravel()[:, None], Y.ravel()[:, None], Z.ravel()[:, None]], axis=1
            )
        else:
            C = np.concatenate([X.ravel()[:, None], Y.ravel()[:, None]], axis=1)

    src_crs = kwargs.get("src_crs", get_default_projection())
    trg_crs = kwargs.get("trg_crs", get_default_projection())
    area_of_interest = kwargs.get("area_of_interest", None)

    axis_order = osr.OAMS_TRADITIONAL_GIS_ORDER
    src_crs.SetAxisMappingStrategy(axis_order)
    trg_crs.SetAxisMappingStrategy(axis_order)
    options = osr.CoordinateTransformationOptions()
    if area_of_interest is not None:
        options.SetAreaOfInterest(*area_of_interest)
    ct = osr.CreateCoordinateTransformation(src_crs, trg_crs, options)
    trans = np.array(ct.TransformPoints(C))

    if len(args) == 1:
        return trans[:, 0:numCols].reshape(cshape)
    else:
        X = trans[:, 0].reshape(xshape)
        Y = trans[:, 1].reshape(yshape)
        if len(args) == 2:
            return X, Y
        if len(args) == 3:
            Z = trans[:, 2].reshape(zshape)
            return X, Y, Z


@reproject.register(DataArray)
@reproject.register(Dataset)
def _reproject_xarray(obj, **kwargs):
    """Transform coordinates from current projection to a target projection.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    trg_crs : :py:class:`gdal:osgeo.osr.SpatialReference`
    coords : dict, optional
        Mapping of coordinates. Defaults to None
    area_of_interest : tuple
        floats (WestLongitudeDeg, SouthLatitudeDeg, EastLongitudeDeg,
        NorthLatitudeDeg), defaults to None (no area of interest).

    Returns
    -------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        reprojected Dataset/DataArray

    Examples
    --------
    See :ref:`/notebooks/georeferencing/georef.ipynb`.
    """
    obj = obj.copy()
    coords = kwargs.pop("coords", None)

    args = []
    if coords is None:
        coords = dict(x="x", y="y", z="z")
    args.append(obj[coords.get("x")].reset_coords(drop=True))
    args.append(obj[coords.get("y")].reset_coords(drop=True))
    if "z" in coords:
        args.append(obj[coords["z"]].reset_coords(drop=True))
    input_core_dims = [list(arg.dims) for arg in args]
    output_core_dims = input_core_dims

    # user overrides?
    if src_crs := kwargs.get("src_crs") is None:
        # extract crs from obj
        proj_crs = xd.georeference.get_crs(obj)
        src_crs = wkt_to_osr(proj_crs.to_wkt())
    else:
        warnings.warn(
            "`src_crs`-kwarg is overriding `crs_wkt`-coordinate'", stacklevel=4
        )

    kwargs.setdefault("src_crs", src_crs)
    trg_crs = kwargs.setdefault("trg_crs", get_default_projection())

    out = apply_ufunc(
        reproject,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )

    for c, v in zip(coords, out):
        obj = obj.assign_coords({c: v})

    # set new crs to obj
    proj_crs = pyproj.CRS.from_wkt(trg_crs.ExportToWkt(["FORMAT=WKT2_2018"]))
    obj = xd.georeference.add_crs(obj, crs=proj_crs)

    return obj


def get_default_projection():
    """Create a default projection object (wgs84)"""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)
    return crs


def epsg_to_osr(epsg=None):
    """Create osr spatial reference object from EPSG number

    Parameters
    ----------
    epsg : int
        EPSG-Number defining the coordinate system

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection
    """
    crs = None
    if epsg:
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(epsg)
    else:
        crs = get_default_projection()
    return crs


def wkt_to_osr(wkt=None):
    """Create osr spatial reference object from WKT string

    Parameters
    ----------
    wkt : str
        WTK string defining the coordinate reference system

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection

    """
    crs = None
    if wkt:
        crs = osr.SpatialReference()
        crs.ImportFromWkt(wkt)
    else:
        crs = get_default_projection()

    if crs.Validate() == ogr.OGRERR_CORRUPT_DATA:
        raise ValueError(
            "wkt validates to 'ogr.OGRERR_CORRUPT_DATA'"
            "and can't be imported as OSR object"
        )

    return crs


@singledispatch
def get_earth_radius(latitude, *, sr=None):
    """Get the radius of the Earth (in km) for a given Spheroid model (sr) at \
    a given position.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    sr : :py:class:`gdal:osgeo.osr.SpatialReference`
        spatial reference
    latitude : float
        geodetic latitude in degrees

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if sr is None:
        sr = get_default_projection()
    radius_e = sr.GetSemiMajor()
    radius_p = sr.GetSemiMinor()
    latitude = np.radians(latitude)
    radius = np.sqrt(
        (
            np.power(radius_e, 4) * np.power(np.cos(latitude), 2)
            + np.power(radius_p, 4) * np.power(np.sin(latitude), 2)
        )
        / (
            np.power(radius_e, 2) * np.power(np.cos(latitude), 2)
            + np.power(radius_p, 2) * np.power(np.sin(latitude), 2)
        )
    )
    return radius


@get_earth_radius.register(Dataset)
@get_earth_radius.register(DataArray)
def _get_earth_radius_xarray(obj, *, sr=None):
    """Get the radius of the Earth (in km) for a given Spheroid model (sr) at \
    a given position.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
    sr : :py:class:`gdal:osgeo.osr.SpatialReference`
        spatial reference

    Returns
    -------
    radius : float
        earth radius in meter
    """
    return get_earth_radius(obj.latitude.values, sr=sr)


def get_earth_projection(model="ellipsoid"):
    """Get a default earth projection based on WGS

    Parameters
    ----------
    model : str
        earth model used, defaults to `ellipsoid`:

        - 'ellipsoid' - WGS84 with ellipsoid heights -> EPSG 4979
        - 'geoid' - WGS84 with egm96 geoid heights -> EPSG 4326 + 5773
        - 'sphere' - GRS 1980 authalic sphere -> EPSG 4047

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection definition

    """
    crs = osr.SpatialReference()

    if model == "sphere":
        crs.ImportFromEPSG(4047)
    elif model == "ellipsoid":
        crs.ImportFromEPSG(4979)
    elif model == "geoid":
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        egm96 = osr.SpatialReference()
        egm96.ImportFromEPSG(5773)
        crs = osr.SpatialReference()
        crs.SetCompoundCS("WGS84 Horizontal + EGM96 Vertical", wgs84, egm96)
    else:
        raise ValueError(f"Unknown model {model!r}.")

    return crs


def get_radar_projection(site):
    """Get the native radar projection which is an azimuthal equidistant projection
    centered at the site using WGS84.

    Parameters
    ----------
    site : sequence
        the WGS84 lon / lat coordinates of the radar location

    Returns
    -------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection definition

    """
    crs = osr.SpatialReference()
    crs.SetProjCS("Unknown Azimuthal Equidistant")
    crs.SetAE(site[1], site[0], 0, 0)

    return crs


def get_extent(coords):
    """Get the extent of 2d coordinates

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        coordinates array with shape (...,(x,y))

    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax)
    """

    xmin = coords[..., 0].min()
    xmax = coords[..., 0].max()
    ymin = coords[..., 1].min()
    ymax = coords[..., 1].max()

    return xmin, xmax, ymin, ymax


class GeorefProjectionMethods:
    """wradlib xarray SubAccessor methods for Georef Projection Methods."""

    @docstring(_get_earth_radius_xarray)
    def get_earth_radius(self, *args, **kwargs):
        if not isinstance(self, GeorefProjectionMethods):
            return get_earth_radius(self, *args, **kwargs)
        else:
            return get_earth_radius(self._obj, *args, **kwargs)

    @docstring(_reproject_xarray)
    def reproject(self, *args, **kwargs):
        if not isinstance(self, GeorefProjectionMethods):
            return reproject(self, *args, **kwargs)
        else:
            return reproject(self._obj, *args, **kwargs)
