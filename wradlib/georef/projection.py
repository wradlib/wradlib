#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
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
    "proj4_to_osr",
    "epsg_to_osr",
    "wkt_to_osr",
    "get_default_projection",
    "get_earth_radius",
    "get_radar_projection",
    "get_earth_projection",
    "get_extent",
]
__doc__ = __doc__.format("\n   ".join(__all__))

from distutils.version import LooseVersion

import numpy as np
from osgeo import gdal, ogr, osr


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
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Azimuthal_Equidistant"],'
        'PARAMETER["latitude_of_center", {0:-f}],'
        'PARAMETER["longitude_of_center", {1:-f}],'
        'PARAMETER["false_easting", {2:-f}],'
        'PARAMETER["false_northing", {3:-f}],'
        'UNIT["Meter",1]]'
    )
    aeqd_wkt3 = (
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

    radolan_wkt3 = (
        'PROJCS["Radolan Projection",'
        'GEOGCS["Radolan Coordinate System",'
        'DATUM["Radolan_Kugel",'
        'SPHEROID["Erdkugel", 6370040, 0]],'
        'PRIMEM["Greenwich", 0,'
        'AUTHORITY["EPSG","8901"]],'
        'UNIT["degree", 0.017453292519943295,'
        'AUTHORITY["EPSG","9122"]]],'
        'PROJECTION["Polar_Stereographic"],'
        'PARAMETER["latitude_of_origin", 90],'
        'PARAMETER["central_meridian", 10],'
        'PARAMETER["scale_factor", {0:8.12f}],'
        'PARAMETER["false_easting", 0],'
        'PARAMETER["false_northing", 0],'
        'UNIT["kilometre", 1000,'
        'AUTHORITY["EPSG","9036"]],'
        'AXIS["Easting",SOUTH],'
        'AXIS["Northing",SOUTH]]'
    )

    radolan_wkt = (
        'PROJCS["Radolan projection",'
        'GEOGCS["Radolan Coordinate System",'
        'DATUM["Radolan Kugel",'
        'SPHEROID["Erdkugel", 6370040.0, 0.0]],'
        'PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],'
        'UNIT["degree", 0.017453292519943295],'
        'AXIS["Longitude", EAST],'
        'AXIS["Latitude", NORTH]],'
        'PROJECTION["polar_stereographic"],'
        'PARAMETER["central_meridian", 10.0],'
        'PARAMETER["latitude_of_origin", 90.0],'
        'PARAMETER["scale_factor", {0:8.10f}],'
        'PARAMETER["false_easting", 0.0],'
        'PARAMETER["false_northing", 0.0],'
        'UNIT["m*1000.0", 1000.0],'
        'AXIS["X", EAST],'
        'AXIS["Y", NORTH]]'
    )

    proj = osr.SpatialReference()

    if projname == "aeqd":
        # Azimuthal Equidistant
        if LooseVersion(gdal.VersionInfo("RELEASE_NAME")) >= LooseVersion("3"):
            aeqd_wkt = aeqd_wkt3

        if "x_0" in kwargs:
            proj.ImportFromWkt(
                aeqd_wkt.format(
                    kwargs["lat_0"], kwargs["lon_0"], kwargs["x_0"], kwargs["y_0"]
                )
            )
        else:
            proj.ImportFromWkt(
                aeqd_wkt.format(kwargs["lat_0"], kwargs["lon_0"], 0.0, 0.0)
            )

    elif projname == "dwd-radolan":
        # DWD-RADOLAN polar stereographic projection
        scale = (1.0 + np.sin(np.radians(60.0))) / (1.0 + np.sin(np.radians(90.0)))
        if LooseVersion(gdal.VersionInfo("RELEASE_NAME")) >= LooseVersion("3"):
            radolan_wkt = radolan_wkt3.format(scale)
        else:
            radolan_wkt = radolan_wkt.format(scale)

        proj.ImportFromWkt(radolan_wkt)
    else:
        raise ValueError(
            "No convenience support for projection %r, "
            "yet.\nYou need to create projection by using "
            "other means..." % projname
        )

    return proj


def proj4_to_osr(proj4str):
    """Transform a proj4 string to an osr spatial reference object

    Parameters
    ----------
    proj4str : str
        Proj4 string describing projection

    Examples
    --------

    See :ref:`/notebooks/radolan/radolan_grid.ipynb#PROJ.4`.

    """
    proj = osr.SpatialReference()
    proj.ImportFromProj4(proj4str)
    proj.AutoIdentifyEPSG()

    if LooseVersion(gdal.VersionInfo("RELEASE_NAME")) < LooseVersion("3"):
        proj.Fixup()
        proj.FixupOrdering()
    if proj.Validate() == ogr.OGRERR_CORRUPT_DATA:
        raise ValueError(
            "proj4str validates to 'ogr.OGRERR_CORRUPT_DATA'"
            "and can't be imported as OSR object"
        )
    return proj


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
    projection_source : :py:class:`gdal:osgeo.osr.SpatialReference`
        defaults to EPSG(4326)
    projection_target : :py:class:`gdal:osgeo.osr.SpatialReference`
        defaults to EPSG(4326)
    area_of_interest : tuple
        tuple of floats (WestLongitudeDeg, SouthLatitudeDeg, EastLongitudeDeg,
        NorthLatitudeDeg), only gdal>=3

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

    See :ref:`/notebooks/georeferencing/wradlib_georef_example.ipynb`.

    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape = C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1, numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError("Input Array column mismatch to %s" % ("reproject"))
    else:
        if len(args) == 2:
            X, Y = (np.asanyarray(arg) for arg in args)
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            numCols = 3
        else:
            raise TypeError("Illegal arguments to %s" % ("reproject"))

        xshape = X.shape
        yshape = Y.shape

        if xshape != yshape:
            raise TypeError("Incompatible X, Y inputs to %s" % ("reproject"))

        if "Z" in locals():
            if xshape != zshape:
                raise TypeError("Incompatible Z input to %s" % ("reproject"))
            C = np.concatenate(
                [X.ravel()[:, None], Y.ravel()[:, None], Z.ravel()[:, None]], axis=1
            )
        else:
            C = np.concatenate([X.ravel()[:, None], Y.ravel()[:, None]], axis=1)

    projection_source = kwargs.get("projection_source", get_default_projection())
    projection_target = kwargs.get("projection_target", get_default_projection())
    area_of_interest = kwargs.get("area_of_interest", None)

    if LooseVersion(gdal.VersionInfo("RELEASE_NAME")) >= LooseVersion("3"):
        axis_order = osr.OAMS_TRADITIONAL_GIS_ORDER
        projection_source.SetAxisMappingStrategy(axis_order)
        projection_target.SetAxisMappingStrategy(axis_order)
        options = osr.CoordinateTransformationOptions()
        if area_of_interest is not None:
            options.SetAreaOfInterest(*area_of_interest)
        ct = osr.CreateCoordinateTransformation(
            projection_source, projection_target, options
        )
    else:
        ct = osr.CoordinateTransformation(projection_source, projection_target)
    trans = np.array(ct.TransformPoints(C))

    if len(args) == 1:
        # here we could do this one
        # return(np.array(ct.TransformPoints(C))[...,0:numCols]))
        # or this one
        trans = trans[:, 0:numCols].reshape(cshape)
        return trans
    else:
        X = trans[:, 0].reshape(xshape)
        Y = trans[:, 1].reshape(yshape)
        if len(args) == 2:
            return X, Y
        if len(args) == 3:
            Z = trans[:, 2].reshape(zshape)
            return X, Y, Z


def get_default_projection():
    """Create a default projection object (wgs84)"""
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    return proj


def epsg_to_osr(epsg=None):
    """Create osr spatial reference object from EPSG number

    Parameters
    ----------
    epsg : int
        EPSG-Number defining the coordinate system

    Returns
    -------
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection
    """
    proj = None
    if epsg:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg)
    else:
        proj = get_default_projection()
    return proj


def wkt_to_osr(wkt=None):
    """Create osr spatial reference object from WKT string

    Parameters
    ----------
    wkt : str
        WTK string defining the coordinate reference system

    Returns
    -------
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection

    """
    proj = None
    if wkt:
        proj = osr.SpatialReference()
        proj.ImportFromWkt(wkt)
    else:
        proj = get_default_projection()

    if proj.Validate() == ogr.OGRERR_CORRUPT_DATA:
        raise ValueError(
            "wkt validates to 'ogr.OGRERR_CORRUPT_DATA'"
            "and can't be imported as OSR object"
        )

    return proj


def get_earth_radius(latitude, sr=None):
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
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection definition

    """
    proj = osr.SpatialReference()

    if model == "sphere":
        proj.ImportFromEPSG(4047)
    elif model == "ellipsoid":
        proj.ImportFromEPSG(4979)
    elif model == "geoid":
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        egm96 = osr.SpatialReference()
        egm96.ImportFromEPSG(5773)
        proj = osr.SpatialReference()
        proj.SetCompoundCS("WGS84 Horizontal + EGM96 Vertical", wgs84, egm96)
    else:
        raise ValueError(f"wradlib: Unknown model='{model}'.")

    return proj


def get_radar_projection(sitecoords):
    """Get the native radar projection which is an azimuthal equidistant projection
    centered at the site using WGS84.

    Parameters
    ----------
    sitecoords : sequence
        the WGS84 lon / lat coordinates of the radar location

    Returns
    -------
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection definition

    """
    proj = osr.SpatialReference()
    proj.SetProjCS("Unknown Azimuthal Equidistant")
    proj.SetAE(sitecoords[1], sitecoords[0], 0, 0)

    return proj


def get_extent(coords):
    """Get the extent of 2d coordinates

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        coordinates array with shape (...,(x,y))

    Returns
    -------
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL/OSR object defining projection
    """

    xmin = coords[..., 0].min()
    xmax = coords[..., 0].max()
    ymin = coords[..., 1].min()
    ymax = coords[..., 1].max()

    return xmin, xmax, ymin, ymax
