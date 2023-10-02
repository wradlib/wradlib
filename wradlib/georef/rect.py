#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Rectangular Grid Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "get_radolan_coords",
    "get_radolan_coordinates",
    "get_radolan_grid",
    "xyz_to_spherical",
    "grid_to_polyvert",
    "GeorefRectMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))
__doctest_requires__ = {"get_radolan_grid": ["osgeo"]}

from functools import singledispatch

import numpy as np
from xarray import DataArray, Dataset, concat

from wradlib.georef import projection
from wradlib.util import docstring, has_import, import_optional

osr = import_optional("osgeo.osr")


def get_radolan_coords(lon, lat, **kwargs):
    """
    Calculates x,y coordinates of radolan grid from lon, lat

    Parameters
    ----------

    lon : float or :class:`numpy:numpy.ndarray`
        longitude
    lat : float, :class:`numpy:numpy.ndarray`
        latitude

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference` | str
        projection of the DWD grid with spheroid model or string `trig` to use
        trigonometric formulas for calculation (only for earth model - `sphere`).
        Defaults to None (earth model - sphere).
    """
    crs = kwargs.get("crs", None)
    # use trig if osgeo.osr is not available
    if crs is None and not has_import(osr):
        crs = "trig"
    if crs == "trig":
        # calculation of x_0 and y_0 coordinates of radolan grid
        # as described in the format description
        phi_0 = np.radians(60)
        phi_m = np.radians(lat)
        lam_0 = 10
        lam_m = lon
        lam = np.radians(lam_m - lam_0)
        er = 6370.040
        m_phi = (1 + np.sin(phi_0)) / (1 + np.sin(phi_m))
        x = er * m_phi * np.cos(phi_m) * np.sin(lam)
        y = -er * m_phi * np.cos(phi_m) * np.cos(lam)
    else:
        # create radolan projection osr object
        if crs is None:
            crs = projection.create_osr("dwd-radolan")

        # create wgs84 projection osr object
        crs_wgs = projection.get_default_projection()

        x, y = projection.reproject(lon, lat, src_crs=crs_wgs, trg_crs=crs)

    return x, y


def get_radolan_coordinates(nrows=None, ncols=None, **kwargs):
    """Calculates x/y coordinates of radolan  projection of the German Weather Service

    Returns the 1D x,y coordinates of the radolan projection for the given grid
    layout.

    Parameters
    ----------
    nrows : int
        number of rows (460, 900 by default, 1100, 1500)
    ncols : int
        number of columns (460, 900 by default, 1400)

    Keyword Arguments
    -----------------
    wgs84 : bool
        if True, output coordinates are in wgs84 lonlat format (default: False)
    mode : str
        'radolan' - lower left pixel coordinates
        'center' - pixel center coordinates
        'edge' - pixel edge coordinates
    crs : :py:class:`gdal:osgeo.osr.SpatialReference` | str
        projection of the DWD grid with spheroid model or string `trig` to use
        trigonometric formulas for calculation (only for earth model - `sphere`).
        Defaults to None (earth model - sphere).

    Returns
    -------
    radolan_ccords : tuple
        x and y 1D coordinate :class:`numpy:numpy.ndarray`
        shape is (nrows, ) and  (ncols, ) if `mode='radolan'`
        shape is (nrows, ) and (ncols, ) if `mode='center'`
        shape is (nrows+1, ) and (ncols+1, ) if `mode='edge'`
    """
    # setup default parameters in dicts
    tiny = {"j_0": 450, "i_0": 450, "res": 2}
    small = {"j_0": 460, "i_0": 460, "res": 2}
    rx = {"j_0": 450, "i_0": 450, "res": 1}
    normal_wx = {"j_0": 370, "i_0": 550, "res": 1}
    de1200 = {"j_0": 470, "i_0": 600, "res": 1}
    extended = {"j_0": 600, "i_0": 800, "res": 1}
    de4800 = {"j_0": 470, "i_0": 600, "res": 0.25}
    griddefs = {
        (450, 450): tiny,
        (460, 460): small,
        (900, 900): rx,
        (1100, 900): normal_wx,
        (1200, 1100): de1200,
        (1500, 1400): extended,
        (4800, 4400): de4800,
    }

    mode = kwargs.get("mode", "radolan")
    crs = kwargs.get("crs", None)
    if nrows and ncols:
        if not (isinstance(nrows, int) and isinstance(ncols, int)):
            raise TypeError("Parameter `nrows` and `ncols` need to be of integer type.")
        if (nrows, ncols) not in griddefs.keys():
            raise ValueError(
                f"Parameter `nrows` ({nrows}) and `ncols` ({ncols}) are no "
                f"valid RADOLAN grid size numbers."
            )
    else:
        # fallback for call without parameters
        nrows = 900
        ncols = 900

    # tiny, small, normal or extended grid check
    # reference point changes according to radolan composit format
    j_0 = griddefs[(nrows, ncols)]["j_0"]
    i_0 = griddefs[(nrows, ncols)]["i_0"]
    res = griddefs[(nrows, ncols)]["res"]

    x_0, y_0 = get_radolan_coords(9.0, 51.0, crs=crs)

    if mode == "edge":
        ncols += 1
        nrows += 1

    # get from km to meter for meter-base projections
    if crs is not None and crs != "trig":
        lin = crs.GetLinearUnits()
        if lin == 1.0:
            res *= 1000.0
            j_0 *= 1000.0
            i_0 *= 1000.0

    x_arr = np.arange(x_0 - j_0, x_0 - j_0 + ncols * res, res)
    y_arr = np.arange(y_0 - i_0, y_0 - i_0 + nrows * res, res)

    if mode == "center":
        x_arr += res / 2.0
        y_arr += res / 2.0

    return x_arr, y_arr


def get_radolan_grid(nrows=None, ncols=None, **kwargs):
    """Calculates x/y coordinates of radolan grid of the German Weather Service

    Returns the x,y coordinates of the radolan grid positions
    (lower left corner of every pixel). The radolan grid is a
    polarstereographic projection, the projection information was taken from
    RADOLAN-RADVOR-OP Kompositformat_2.2.2  :cite:`DWD2009`

    .. table:: Coordinates for 900km x 900km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  3.5889E  |  46.9526N  | -523.4622 | -4658.645 |
        +------------+-----------+------------+-----------+-----------+
        | LowerRight | 14.6209E  |  47.0705N  |  376.5378 | -4658.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperRight | 15.7208E  |  54.7405N  |  376.5378 | -3758.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperLeft  |  2.0715E  |  54.5877N  | -523.4622 | -3758.645 |
        +------------+-----------+------------+-----------+-----------+

    .. table:: Coordinates for 1100km x 900km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  4.6759E  |  46.1929N  | -443.4622 | -4758.645 |
        +------------+-----------+------------+-----------+-----------+
        | LowerRight | 15.4801E  |  46.1827N  |  456.5378 | -4758.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperRight | 17.1128E  |  55.5342N  |  456.5378 | -3658.645 |
        +------------+-----------+------------+-----------+-----------+
        | UpperLeft  |  3.0889E  |  55.5482N  | -443.4622 | -3658.645 |
        +------------+-----------+------------+-----------+-----------+

    .. table:: Coordinates for 1500km x 1400km grid

        +------------+-----------+------------+-----------+-----------+
        | Coordinate |   lon     |     lat    |     x     |     y     |
        +============+===========+============+===========+===========+
        | LowerLeft  |  2.3419E  |  43.9336N  | -673.4622 | -5008.645 |
        +------------+-----------+------------+-----------+-----------+

    Parameters
    ----------
    nrows : int
        number of rows (460, 900 by default, 1100, 1500)
    ncols : int
        number of columns (460, 900 by default, 1400)

    Keyword Arguments
    -----------------
    wgs84 : bool
        if True, output coordinates are in wgs84 lonlat format (default: False)
    mode :  str
        'radolan' - lower left pixel coordinates
        'center' - pixel center coordinates
        'edge' - pixel edge coordinates
    crs : :py:class:`gdal:osgeo.osr.SpatialReference` | str
        projection of the DWD grid with spheroid model or string `trig` to use
        trigonometric formulas for calculation (only for earth model - `sphere`).
        Defaults to None (earth model - sphere).

    Returns
    -------
    radolan_grid : :class:`numpy:numpy.ndarray`
        Array of xy- or lonlat-grid.
        shape is (nrows, ncols, 2) if `mode='radolan'`
        shape is (nrows, ncols, 2) if `mode='center'`
        shape is (nrows+1, ncols+1, 2) if `mode='edge'`

    Examples
    --------

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef  # noqa
    >>> radolan_grid = georef.get_radolan_grid()
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (-523.4622, -4658.6447)

    >>> # using pure trigonometric transformations
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(crs="trig")
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (-523.4622, -4658.6447)

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(1500, 1400)
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (1500, 1400, 2), (-673.4622, -5008.6447)

    >>> # using osr spatial reference transformation
    >>> import wradlib.georef as georef
    >>> radolan_grid = georef.get_radolan_grid(900, 900, wgs84=True)
    >>> print("{0}, ({1:.4f}, {2:.4f})".format(radolan_grid.shape, *radolan_grid[0,0,:]))  # noqa
    (900, 900, 2), (3.5889, 46.9526)

    See :ref:`/notebooks/fileio/radolan/radolan_grid.ipynb#Polar-Stereographic-Projection`.

    Raises
    ------
        TypeError, ValueError
    """

    wgs84 = kwargs.get("wgs84", False)
    mode = kwargs.get("mode", "radolan")
    crs = kwargs.get("crs", None)

    # fallback to simple trignometry, if GDAL is not installed
    if crs is None and not has_import(osr):
        crs = "trig"

    x_arr, y_arr = get_radolan_coordinates(nrows=nrows, ncols=ncols, mode=mode, crs=crs)

    x, y = np.meshgrid(x_arr, y_arr)

    radolan_grid = np.dstack((x, y))

    if wgs84:
        if crs == "trig":
            # inverse projection
            lon0 = 10.0  # central meridian of projection
            lat0 = 60.0  # standard parallel of projection

            sinlat0 = np.sin(np.radians(lat0))

            fac = (6370.040**2.0) * ((1.0 + sinlat0) ** 2.0)
            lon = np.degrees(np.arctan(-x / y)) + lon0
            lat = np.degrees(
                np.arcsin((fac - (x**2.0 + y**2.0)) / (fac + (x**2.0 + y**2.0)))
            )
            radolan_grid = np.dstack((lon, lat))
        else:
            # create radolan projection osr object
            if crs is None:
                crs = projection.create_osr("dwd-radolan")

            # create wgs84 projection osr object
            crs_wgs84 = projection.get_default_projection()

            radolan_grid = projection.reproject(
                radolan_grid, src_crs=crs, trg_crs=crs_wgs84
            )

    return radolan_grid


@singledispatch
def xyz_to_spherical(*args, **kwargs):
    pass


@xyz_to_spherical.register(np.ndarray)
def _xyz_to_spherical_numpy(xyz, *, altitude=0, crs=None, ke=4.0 / 3.0):
    """Returns spherical representation (r, theta, phi) of given cartesian
    coordinates (x, y, z) with respect to the reference altitude (asl)
    considering earth's geometry (crs).

    Parameters
    ----------
    xyz : :class:`numpy:numpy.ndarray`
        Array of shape (..., 3). Contains cartesian coordinates.
    altitude : float
        Altitude (in meters)
        defaults to 0.
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection of the source coordinates (aeqd) with spheroid model
        defaults to None.
    ke : float
        Adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    r : :class:`numpy:numpy.ndarray`
        Array of xyz.shape. Contains the radial distances.
    phi : :class:`numpy:numpy.ndarray`
        Array of xyz.shape. Contains the azimuthal angles.
    theta: :class:`numpy:numpy.ndarray`
        Array of xyz.shape. Contains the elevation angles.
    """

    # get the approximate radius of the projection's ellipsoid
    # for the latitude_of_center, if no projection is given assume
    # spherical earth
    try:
        lat0 = crs.GetProjParm("latitude_of_center")
        re = projection.get_earth_radius(lat0, crs)
    except Exception:
        re = 6371000.0

    # calculate xy-distance
    s = np.sqrt(np.sum(xyz[..., 0:2] ** 2, axis=-1))

    # calculate earth's arc angle
    gamma = s / (re * ke)

    # calculate elevation angle theta
    numer = np.cos(gamma) - (re * ke + altitude) / (re * ke + xyz[..., 2])
    denom = np.sin(gamma)
    theta = np.arctan(numer / denom)

    # calculate radial distance r
    r = (re * ke + xyz[..., 2]) * denom / np.cos(theta)

    # calculate azimuth angle phi
    phi = np.degrees(np.arctan2(xyz[..., 0], xyz[..., 1]))
    phi = np.fmod(phi + 360, 360)

    return r, phi, np.degrees(theta)


@xyz_to_spherical.register(Dataset)
@xyz_to_spherical.register(DataArray)
def _xyz_to_spherical_xarray(obj, **kwargs):
    """Returns spherical representation (r, theta, phi) of given cartesian
    coordinates (x, y, z) with respect to the reference altitude (asl)
    considering earth's geometry (crs).

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        projection of the source coordinates (aeqd) with spheroid model
        defaults to None.
    ke : float
        Adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependend. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    obj : :py:class:`xarray:xarray.Dataset`
        obj with added spherical coordinates.
    """
    # transform xp,yp,zp to ncoord
    xyzp = concat([obj.xp, obj.yp, obj.zp], dim="ncoord").transpose(..., "ncoord")

    r_sr, az_sr, elev_sr = _xyz_to_spherical_numpy(
        xyzp, altitude=xyzp.altitude, **kwargs
    )
    obj = obj.assign_coords(
        {
            "range": r_sr,
            "azimuth": az_sr,
            "elevation": elev_sr,
        }
    )
    return obj


def grid_to_polyvert(grid, *, ravel=False):
    """Get polygonal vertices from rectangular grid coordinates.

    Parameters
    ----------
    grid : :class:`numpy:numpy.ndarray`
        grid edge coordinates
    ravel : bool, optional
        option to flatten the grid, defaults to False.

    Returns
    -------
    polyvert : :class:`numpy:numpy.ndarray`
        A 3-d array of polygon vertices with shape (..., 5, 2).

    """

    v1 = grid[:-1, :-1]
    v2 = grid[:-1, 1:]
    v3 = grid[1:, 1:]
    v4 = grid[1:, :-1]

    polyvert = np.stack((v1, v2, v3, v4, v1), axis=-2)

    if ravel:
        polyvert = polyvert.reshape((-1, 5, 2))

    return polyvert


class GeorefRectMethods:
    """wradlib xarray SubAccessor methods for Georef Rect Methods."""

    @docstring(_xyz_to_spherical_xarray)
    def xyz_to_spherical(self, *args, **kwargs):
        if not isinstance(self, GeorefRectMethods):
            return xyz_to_spherical(self, *args, **kwargs)
        else:
            return xyz_to_spherical(self._obj, *args, **kwargs)
