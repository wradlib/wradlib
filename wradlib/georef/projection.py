#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Projection Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   reproject
   create_osr
   proj4_to_osr
   epsg_to_osr
   wkt_to_osr
"""

from osgeo import osr
import numpy as np
from sys import exit


def create_osr(projname, **kwargs):
    """Conveniently supports the construction of osr spatial reference objects

    .. versionadded:: 0.6.0

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
    projname : string
        "aeqd" or "dwd-radolan"
    kwargs : depends on projname - see above!

    Returns
    -------
    output : osr.SpatialReference
        GDAL/OSR object defining projection

    Examples
    --------

    See :ref:`notebooks/basics/wradlib_workflow.ipynb#\
Georeferencing-and-Projection`.

    """

    aeqd_wkt = ('PROJCS["unnamed",'
                'GEOGCS["WGS 84",'
                'DATUM["unknown",'
                'SPHEROID["WGS84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                'PROJECTION["Azimuthal_Equidistant"],'
                'PARAMETER["latitude_of_center", {0:-f}],'
                'PARAMETER["longitude_of_center", {1:-f}],'
                'PARAMETER["false_easting", {2:-f}],'
                'PARAMETER["false_northing", {3:-f}]]')

    radolan_wkt = ('PROJCS["Radolan projection",'
                   'GEOGCS["Radolan Coordinate System",'
                   'DATUM["Radolan Kugel",'
                   'SPHEROID["Erdkugel", 6370040.0, 0.0]],'
                   'PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],'
                   'UNIT["degree", 0.017453292519943295],'
                   'AXIS["Longitude", EAST],'
                   'AXIS["Latitude", NORTH]],'
                   'PROJECTION["polar_stereographic"],'
                   'PARAMETER["central_meridian", 10.0],'
                   'PARAMETER["latitude_of_origin", 60.0],'
                   'PARAMETER["scale_factor", {0:8.10f}],'
                   'PARAMETER["false_easting", 0.0],'
                   'PARAMETER["false_northing", 0.0],'
                   'UNIT["m*1000.0", 1000.0],'
                   'AXIS["X", EAST],'
                   'AXIS["Y", NORTH]]')
    #                  'AUTHORITY["USER","100000"]]'

    proj = osr.SpatialReference()

    if projname == "aeqd":
        # Azimuthal Equidistant
        if "x_0" in kwargs:
            proj.ImportFromWkt(aeqd_wkt.format(kwargs["lat_0"],
                                               kwargs["lon_0"],
                                               kwargs["x_0"],
                                               kwargs["y_0"]))
        else:
            proj.ImportFromWkt(aeqd_wkt.format(kwargs["lat_0"],
                                               kwargs["lon_0"], 0, 0))

    elif projname == "dwd-radolan":
        # DWD-RADOLAN polar stereographic projection
        scale = (1. + np.sin(np.radians(60.))) / (1. + np.sin(np.radians(90.)))
        proj.ImportFromWkt(radolan_wkt.format(scale))

    else:
        print("No convenience support for projection %r, yet." % projname)
        print("You need to create projection by using other means...")
        exit(1)

    return proj


def proj4_to_osr(proj4str):
    """Transform a proj4 string to an osr spatial reference object

    Parameters
    ----------
    proj4str : string
        Proj4 string describing projection

    Examples
    --------

    See :ref:`notebooks/radolan/radolan_grid.ipynb#PROJ.4`.

    """
    proj = None
    if proj4str:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(proj4str)
    else:
        proj = get_default_projection()
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
    C : multidimensional :class:`numpy:numpy.ndarray`
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
    projection_source : osr object
        defaults to EPSG(4326)
    projection_target : osr object
        defaults to EPSG(4326)

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

    See :ref:`notebooks/georeferencing/wradlib_georef_example.ipynb`.

    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape = C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1, numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError('Input Array column mismatch '
                            'to %s' % ('reproject'))
    else:
        if len(args) == 2:
            X, Y = (np.asanyarray(arg) for arg in args)
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            numCols = 3
        else:
            raise TypeError('Illegal arguments to %s' % ('reproject'))

        xshape = X.shape
        yshape = Y.shape

        if xshape != yshape:
            raise TypeError('Incompatible X, Y inputs to %s' % ('reproject'))

        if 'Z' in locals():
            if xshape != zshape:
                raise TypeError('Incompatible Z input to %s' % ('reproject'))
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None],
                                Z.ravel()[:, None]], axis=1)
        else:
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None]], axis=1)

    projection_source = kwargs.get('projection_source',
                                   get_default_projection())
    projection_target = kwargs.get('projection_target',
                                   get_default_projection())

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

    .. versionadded:: 0.6.0

    Parameters
    ----------
    epsg : int
        EPSG-Number defining the coordinate system

    Returns
    -------
    proj : osr.SpatialReference
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

    .. versionadded:: 0.6.0

    Parameters
    ----------
    wkt : str
        WTK string defining the coordinate reference system

    Returns
    -------
    proj : osr.SpatialReference
        GDAL/OSR object defining projection

    """
    proj = None
    if wkt:
        proj = osr.SpatialReference()
        proj.ImportFromWkt(wkt)
    else:
        proj = get_default_projection()
    return proj
