#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2025, wradlib developers.
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
    "ensure_crs",
    "reproject",
    "create_crs",
    "create_osr",
    "projstr_to_osr",
    "epsg_to_osr",
    "wkt_to_osr",
    "get_default_projection",
    "get_earth_radius",
    "get_radar_projection",
    "get_earth_projection",
    "get_extent",
    "project_bounds",
    "meters_to_degrees",
    "GeorefProjectionMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import re
import warnings
from functools import singledispatch

import deprecation
import numpy as np
import pyproj
import xradar as xd
from xarray import DataArray, Dataset, apply_ufunc

from wradlib import version
from wradlib.util import docstring, has_import, import_optional

gdal = import_optional("osgeo.gdal")
ogr = import_optional("osgeo.ogr")
osr = import_optional("osgeo.osr")
cartopy = import_optional("cartopy")

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


def ensure_crs(crs, trg="pyproj"):
    """Return CRS object from given entry. Default to pyproj CRS.

    Parameters
    ----------
    crs
        Coordinate Reference System (CRS) of the coordinates. Must be given and
        can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.crs.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)
        - None

    Keyword Arguments
    -----------------
    trg : str, {"pyproj", "cartopy", "osr"}
        Target Coordinate Reference System type

    Returns
    -------
    crs : :py:class:`pyproj:pyproj.crs.CoordinateSystem`, :py:class:`cartopy:cartopy.crs.CRS`, :py:class:`gdal:osgeo.osr.SpatialReference` or None
    """
    # first move everything into pyproj.CRS/WKT or return early
    if crs is None:
        return crs
    if isinstance(crs, pyproj.CRS) and type(crs) is pyproj.CRS:
        if trg == "pyproj":
            return crs
    elif has_import(cartopy) and isinstance(crs, cartopy.crs.CRS):
        if trg == "cartopy":
            return crs
    elif has_import(osr) and isinstance(crs, osr.SpatialReference):
        if trg == "osr":
            return crs
        crs = crs.ExportToWkt()

    # now ingest into pyproj.CRS
    crs = pyproj.CRS.from_user_input(crs)

    if trg == "pyproj":
        return crs
    elif trg == "cartopy":
        return cartopy.crs.CRS(crs)
    elif trg == "osr":
        out = osr.SpatialReference()
        out.ImportFromWkt(crs.to_wkt())
        return out


@deprecation.deprecated(
    deprecated_in="2.4",
    removed_in="3.0",
    current_version=version.version,
    details="Use `wradlib.georef.projection.create_crs` instead.",
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
    See :ref:`notebooks:notebooks/basics/wradlib_workflow:georeferencing and projection`.
    """
    crs = osr.SpatialReference()
    crs.ImportFromWkt(create_crs(projname, **kwargs).to_wkt())
    return crs


def create_crs(projname, **kwargs):
    """Conveniently supports the construction of pyproj Coordinate Reference System (CRS)

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
    output : :py:class:`pyproj:pyproj.crs.CoordinateSystem`
        pyproj Coordinate Reference System (CRS)

    Examples
    --------
    See :ref:`notebooks:notebooks/basics/wradlib_workflow:georeferencing and projection`.
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
        'DATUM["Unknown based on WGS 84 ellipsoid",'
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

    if projname == "aeqd":
        # Azimuthal Equidistant
        x_0 = kwargs.get("x_0", 0.0)
        y_0 = kwargs.get("y_0", 0.0)
        if "x_0" in kwargs:
            crs = pyproj.CRS.from_wkt(
                aeqd_wkt.format(kwargs["lat_0"], kwargs["lon_0"], x_0, y_0)
            )
        else:
            crs = pyproj.CRS.from_wkt(
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
        crs = pyproj.CRS.from_wkt(radolan_wkt)
    else:
        raise ValueError(
            f"No convenience support for projection {projname!r} yet. "
            f"Please create projection by using other means."
        )

    return crs


@deprecation.deprecated(
    deprecated_in="2.4",
    removed_in="3.0",
    current_version=version.version,
    details="Use `pyproj.CRS.from_proj4` instead.",
)
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
    See :ref:`radolan:projection`.
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
    src_crs
        Coordinate Reference System (CRS) of the coordinates. Can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to EPSG(4326).
    trg_crs
        Coordinate Reference System (CRS) of the coordinates. Can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to EPSG(4326).
    area_of_interest : tuple
        floats (WestLongitudeDeg, SouthLatitudeDeg, EastLongitudeDeg,
        NorthLatitudeDeg), defaults to None (no area of interest)

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
    See :doc:`notebooks:notebooks/georeferencing/georef`.
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
            C = np.column_stack([X.ravel(), Y.ravel()])
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            C = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
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

    src_crs = kwargs.get("src_crs", get_default_projection())
    src_crs = ensure_crs(src_crs)
    trg_crs = kwargs.get("trg_crs", get_default_projection())
    trg_crs = ensure_crs(trg_crs)

    area_of_interest = kwargs.get("area_of_interest", None)
    if isinstance(area_of_interest, tuple):
        area_of_interest = pyproj.transformer.AreaOfInterest(*area_of_interest)

    # Transformer setup
    transformer = pyproj.Transformer.from_crs(
        src_crs,
        trg_crs,
        always_xy=True,  # like OAMS_TRADITIONAL_GIS_ORDER
        area_of_interest=area_of_interest,
    )

    # --- Transform coordinates ---
    if numCols == 2:
        x, y = transformer.transform(C[:, 0], C[:, 1])
        trans = np.column_stack([x, y])
    else:
        x, y, z = transformer.transform(C[:, 0], C[:, 1], C[:, 2])
        trans = np.column_stack([x, y, z])

    if len(args) == 1:
        return trans.reshape(cshape)
    else:
        out = (x.reshape(xshape), y.reshape(yshape))
        if numCols == 2:
            return out
        if numCols == 3:
            return out + (z.reshape(zshape),)


@reproject.register(DataArray)
@reproject.register(Dataset)
def _reproject_xarray(obj, **kwargs):
    """Transform coordinates from current projection to a target projection.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    src_crs
        Coordinate Reference System (CRS) of the coordinates. Can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to source data CRS.
    trg_crs
        Coordinate Reference System (CRS) of the coordinates. Can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to WGS84.
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
    See :doc:`notebooks:notebooks/georeferencing/georef`.
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
    if (src_crs := kwargs.get("src_crs")) is None:
        # extract crs from obj
        src_crs = xd.georeference.get_crs(obj)
    else:
        warnings.warn(
            "`src_crs`-kwarg is overriding `crs_wkt`-coordinate'", stacklevel=4
        )

    src_crs = ensure_crs(src_crs)
    kwargs["src_crs"] = src_crs

    if (trg_crs := kwargs.get("trg_crs")) is None:
        trg_crs = get_default_projection()
    trg_crs = ensure_crs(trg_crs)
    kwargs["trg_crs"] = trg_crs

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

    # set target crs to obj
    obj = xd.georeference.add_crs(obj, crs=trg_crs)

    return obj


def get_default_projection():
    """Create a default projection object (wgs84)"""
    crs = pyproj.CRS.from_epsg(4326)
    return crs


@deprecation.deprecated(
    deprecated_in="2.4",
    removed_in="3.0",
    current_version=version.version,
    details="Use `pyproj.CRS.from_epsg` instead.",
)
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
        crs = osr.SpatialReference()
        crs.ImportFromWkt(get_default_projection().to_wkt())
    return crs


@deprecation.deprecated(
    deprecated_in="2.4",
    removed_in="3.0",
    current_version=version.version,
    details="Use `pyproj.CRS.from_wkt` instead.",
)
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
        crs = osr.SpatialReference()
        crs.ImportFromWkt(get_default_projection().to_wkt())

    if crs.Validate() == ogr.OGRERR_CORRUPT_DATA:
        raise ValueError(
            "wkt validates to 'ogr.OGRERR_CORRUPT_DATA'"
            "and can't be imported as OSR object"
        )

    return crs


@singledispatch
def get_earth_radius(latitude, *, crs=None):
    """Get the radius of the Earth (in km) for a given Spheroid model (sr) at \
    a given position.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    latitude : float
        geodetic latitude in degrees

    Keyword Arguments
    -----------------
    crs
        Coordinate Reference System (CRS) of the coordinates. Must be provided
        and can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.crs.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to EPSG(4326).

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if crs is None:
        crs = get_default_projection()
    crs = ensure_crs(crs)

    radius_e = crs.ellipsoid.semi_major_metre
    radius_p = crs.ellipsoid.semi_minor_metre
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
def _get_earth_radius_xarray(obj, *, crs=None):
    """Get the radius of the Earth (in km) for a given Spheroid model (sr) at \
    a given position.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    crs
        Coordinate Reference System (CRS) of the coordinates. Can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.crs.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

        Defaults to EPSG(4326).

    Returns
    -------
    radius : float
        earth radius in meter
    """
    return get_earth_radius(obj.latitude.values, crs=crs)


def get_earth_projection(model="ellipsoid", arcsecond=False):
    """Get a default earth projection based on WGS

    Parameters
    ----------
    model : str
        earth model used, defaults to `ellipsoid`:

        - 'ellipsoid' - WGS84 with ellipsoid heights -> EPSG 4979
        - 'geoid' - WGS84 with egm96 geoid heights -> EPSG 4326 + 5773
        - 'sphere' - GRS 1980 authalic sphere -> EPSG 4047
    arcsecond : boolean
        true to use arcsecond as unit instead of degree, defaults to False

    Returns
    -------
    crs : :py:class:`pyproj:pyproj.crs.CoordinateSystem`
        Coordinate Reference System (CRS)

    """
    if model == "sphere":
        crs = pyproj.CRS.from_epsg(4047)
    elif model == "ellipsoid":
        crs = pyproj.CRS.from_epsg(4979)
    elif model == "geoid":
        crs = pyproj.CRS.from_user_input("EPSG:4326+5773")

    if arcsecond:
        wkt = crs.to_wkt()
        wkt = re.sub(
            r"ANGLEUNIT\[[^\]]*\]",
            'ANGLEUNIT["arc-second",4.84813681109536E-06,ID["EPSG",9104]]',
            wkt,
        )
        crs = pyproj.CRS.from_wkt(wkt)

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
    crs : :py:class:`pyproj:pyproj.crs.CoordinateSystem`
        Coordinate Reference System (CRS) - radar centric AEQD

    """
    lon, lat = site[:2]
    projstr = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84"
    crs = pyproj.CRS.from_proj4(projstr)
    wkt = crs.to_wkt(version="WKT2_2018")
    wkt_named = wkt.replace(
        'PROJCRS["unknown"', 'PROJCRS["Unknown Azimuthal Equidistant"'
    )
    crs = pyproj.CRS.from_wkt(wkt_named)

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


def project_bounds(bounds, crs):
    """Get geographic bounds in projected coordinate system

    Parameters
    ----------
    bounds : tuple of float
        (lon_min, lon_max, lat_min, lat_max) geographic bounds
    crs
        Coordinate Reference System (CRS) to be used for projection.
        Must be provided and can be one of:

        - A :py:class:`pyproj:pyproj.crs.CoordinateSystem` instance
        - A :py:class:`cartopy:cartopy.crs.CRS` instance
        - A :py:class:`gdal:osgeo.osr.SpatialReference` instance
        - A type accepted by :py:meth:`pyproj.crs.CRS.from_user_input` (e.g., EPSG code,
          PROJ string, dictionary, WKT, or any object with a `to_wkt()` method)

    Returns
    -------
    bounds : tuople of float
        (xmin, xmax, ymin, ymax) projected bounds

    """
    crs = ensure_crs(crs)

    lon_min, lon_max, lat_min, lat_max = bounds
    lon_mid = lon_min / 2 + lon_max / 2
    lat_mid = lat_min / 2 + lat_max / 2
    (xmin, temp) = reproject((lon_min, lat_mid), trg_crs=crs)
    (temp, ymin) = reproject((lon_mid, lat_min), trg_crs=crs)
    (xmax, temp) = reproject((lon_max, lat_mid), trg_crs=crs)
    (temp, ymax) = reproject((lon_mid, lat_max), trg_crs=crs)
    projected_bounds = (xmin, xmax, ymin, ymax)

    return projected_bounds


def meters_to_degrees(meters, longitude=0.0, latitude=0.0):
    """
    Converts a distance in meters to degrees of latitude and longitude
    using the WGS84 ellipsoid. If scalar, assumes equal east/north offset.

    Parameters
    ----------
    meters :  float or tuple(float, float)
        Distance in meters.
            - If scalar: interpreted as [meters, meters] (diagonal NE).
            - If 2D: interpreted as [east, north] in meters.
    latitude : float
        Reference latitude in degrees.
    longitude : float
        Reference longitude in degrees.

    Returns
    -------
    tuple
        (delta_latitude, delta_longitude) in degrees
    """
    geod = pyproj.Geod(ellps="WGS84")

    # Promote scalar to 2D vector: (east, north)
    if np.isscalar(meters):
        meters = (meters, meters)

    dx, dy = meters
    _, lat1, _ = geod.fwd(longitude, latitude, 0, dy)  # North
    lon1, _, _ = geod.fwd(longitude, latitude, 90, dx)  # East

    delta_lat = lat1 - latitude
    delta_lon = lon1 - longitude

    return delta_lon, delta_lat


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
