#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   beam_height_n
   arc_distance_n
   get_earth_radius
"""

import numpy as np

from .projection import get_default_projection

# Seitenlänge Zenit - Himmelsnordpol: 90°-phi
# Seitenlänge Himmelsnordpol - Gestirn: 90°-delta
# Seitenlänge Zenit - Gestirn: 90°-h
# Winkel Himmelsnordpol - Zenit - Gestirn: 180°-a
# Winkel Zenit - Himmelsnordpol - Gestirn: tau

# alpha - rektaszension
# delta - deklination
# theta - sternzeit
# tau = theta - alpha - stundenwinkel
# a - azimuth (von süden aus gezählt)
# h - Höhe über Horizont


def hor2aeq(a, h, phi):
    """"""
    delta = np.arcsin(np.sin(h) * np.sin(phi) - np.cos(h) *
                      np.cos(a) * np.cos(phi))
    tau = np.arcsin(np.cos(h) * np.sin(a) / np.cos(delta))
    return delta, tau


def aeq2hor(tau, delta, phi):
    """"""
    h = np.arcsin(np.cos(delta) * np.cos(tau) * np.cos(phi) +
                  np.sin(delta) * np.sin(phi))
    a = np.arcsin(np.cos(delta) * np.sin(tau) / np.cos(h))
    return a, h


def beam_height_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the height of a radar beam taking the refractivity of the
    atmosphere into account.

    Based on :cite:`Doviak1993` the beam height is calculated as

    .. math::

        h = \sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \sin\theta} - k_e r_e


    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    height : float
        height of the beam in [m]

    """
    return np.sqrt(r ** 2 + (ke * re) ** 2 +
                   2 * r * ke * re * np.sin(np.radians(theta))) - ke * re


def arc_distance_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the great circle distance of a radar beam over a sphere,
    taking the refractivity of the atmosphere into account.

    Based on :cite:`Doviak1993` the arc distance may be calculated as

    .. math::

        s = k_e r_e \arcsin\left(
        \frac{r \cos\theta}{k_e r_e + h_n(r, \theta, r_e, k_e)}\right)

    where :math:`h_n` would be provided by
    :meth:`~wradlib.georef.beam_height_n`

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : float
        great circle arc distance [m]

    See Also
    --------
    beam_height_n

    """
    return ke * re * np.arcsin((r * np.cos(np.radians(theta))) /
                               (ke * re + beam_height_n(r, theta, re, ke)))


def get_earth_radius(latitude, sr=None):
    r"""
    Get the radius of the Earth (in km) for a given Spheroid model (sr) at a
    given position

    .. math::

        R^2 = \frac{a^4 \cos(f)^2 + b^4 \sin(f)^2}
        {a^2 \cos(f)^2 + b^2 \sin(f)^2}

    Parameters
    ----------
    sr : osr object
        spatial reference;
    latitude : float
        geodetic latitude in degrees;

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if sr is None:
        sr = get_default_projection()
    RADIUS_E = sr.GetSemiMajor()
    RADIUS_P = sr.GetSemiMinor()
    latitude = np.radians(latitude)
    radius = np.sqrt((np.power(RADIUS_E, 4) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 4) * np.power(np.sin(latitude), 2)) /
                     (np.power(RADIUS_E, 2) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 2) * np.power(np.sin(latitude), 2)))
    return (radius)


def get_shape_points(geom):
    """
    Extract coordinate points from given ogr geometry as generator object

    If geometries are nested, function recurses.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    geom : ogr.Geometry

    Returns
    -------
    result : generator object
        expands to Nx2 dimensional nested point arrays
    """

    type = geom.GetGeometryType()
    if type:
        # 1D Geometries, LINESTRINGS
        if type == 2:
            result = np.array(geom.GetPoints())
            yield result
        # RINGS, POLYGONS, MULTIPOLYGONS, MULTILINESTRINGS
        elif type > 2:
            # iterate over geometries and recurse
            for item in geom:
                for result in get_shape_points(item):
                    yield result
    else:
        print("Unknown Geometry")


def transform_geometry(geom, dest_srs):
    """
    Perform geotransformation to given destination SpatialReferenceSystem

    It transforms coordinates to a given destination osr spatial reference
    if a geotransform is neccessary.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    geom : ogr.geometry
    dest_srs: osr.SpatialReference
        Destination Projection

    Returns
    -------
    geom : ogr.Geometry
        Transformed Geometry
    """

    # transform if not the same spatial reference system
    if not geom.GetSpatialReference().IsSame(dest_srs):
        geom.TransformTo(dest_srs)

    return geom


def get_shape_coordinates(layer, **kwargs):
    """
    Function iterates over gdal ogr layer features and packs extracted shape
    coordinate points into nested ndarray

    It transforms coordinates to a given destination osr spatial reference if
    dest_srs is given and a geotransform is neccessary.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    layer : ogr.Layer

    Keywords
    --------
    dest_srs: osr.SpatialReference
        Destination Projection
    key : string
        attribute key to extract from layer feature

    Returns
    -------
    shp : nested :class:`numpy:numpy.ndarray`
        Dimension of subarrays Nx2
        extracted shape coordinate points
    attrs : list
        List of attributes extracted from features
    """

    shp = []

    dest_srs = kwargs.get('dest_srs', None)
    key = kwargs.get('key', None)
    if key:
        attrs = []
    else:
        attrs = None

    for i in range(layer.GetFeatureCount()):
        feature = layer.GetNextFeature()
        if feature:
            if key:
                attrs.append(feature[key])
            geom = feature.GetGeometryRef()
            if dest_srs:
                transform_geometry(geom, dest_srs)
            # get list of xy-coordinates
            reslist = list(get_shape_points(geom))
            shp.append(np.squeeze(np.array(reslist)))

    shp = np.squeeze(np.array(shp))

    return shp, attrs
